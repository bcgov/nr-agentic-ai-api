"""Azure integration helpers used by the agent backend."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..core.config import Settings
from .schema_loader import SchemaLoader

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CosmosClient = None  # type: ignore[assignment]
    PartitionKey = None  # type: ignore[assignment]


@dataclass
class RetrievalChunk:
    id: str
    content: str
    source: str


class RetrievalAugmentor:
    """Perform hybrid retrieval against Azure AI Search with schema fallback."""

    def __init__(self, settings: Settings, schema_loader: SchemaLoader) -> None:
        self.settings = settings
        self.schema_loader = schema_loader
        self._local_index = self._build_local_index()
        self._search_client: Optional[SearchClient] = None

    def _build_local_index(self) -> List[RetrievalChunk]:
        chunks: List[RetrievalChunk] = []
        for schema_file in self.schema_loader.list_schema_files():
            with open(schema_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            page_id = payload.get("id") or schema_file.stem
            for field in payload.get("fields", []):
                chunks.append(
                    RetrievalChunk(
                        id=f"{page_id}::{field['id']}",
                        content=json.dumps(field, indent=2),
                        source=schema_file.name,
                    )
                )
            for popup in payload.get("popups", []):
                chunks.append(
                    RetrievalChunk(
                        id=f"{page_id}::popup::{popup['id']}",
                        content=json.dumps(popup, indent=2),
                        source=schema_file.name,
                    )
                )
        return chunks

    def _get_search_client(self) -> Optional[SearchClient]:
        if self._search_client is not None:
            return self._search_client
        api_key = self.settings.azure_search_api_key
        credential: AzureKeyCredential | DefaultAzureCredential
        if api_key:
            credential = AzureKeyCredential(api_key)
        else:  # pragma: no cover - requires Azure auth context
            credential = DefaultAzureCredential()
        try:
            self._search_client = SearchClient(
                endpoint=str(self.settings.azure_search_endpoint),
                index_name=self.settings.azure_search_index,
                credential=credential,
            )
        except Exception as exc:  # pragma: no cover - environment issues
            LOGGER.warning("Failed to initialise SearchClient: %s", exc)
            self._search_client = None
        return self._search_client

    async def search(self, query: str, page_id: Optional[str]) -> List[RetrievalChunk]:
        if not query:
            return []
        client = self._get_search_client()
        if client:
            try:
                filter_expression = f"pageId eq '{page_id}'" if page_id else None
                vector = await self._compute_vector(query)
                kwargs: Dict[str, Any] = {"search_text": query or "*", "top": 5}
                if filter_expression:
                    kwargs["filter"] = filter_expression
                if vector is not None:
                    kwargs["vector"] = {
                        "value": vector,
                        "k": 5,
                        "fields": "contentVector",
                    }
                results = client.search(**kwargs)  # type: ignore[arg-type]
                chunks: List[RetrievalChunk] = []
                for doc in results:  # pragma: no branch - streaming iterator
                    chunks.append(
                        RetrievalChunk(
                            id=str(doc.get("id", "azure-search")),
                            content=str(doc.get("content", "")),
                            source=str(doc.get("source", self.settings.azure_search_index)),
                        )
                    )
                if chunks:
                    return chunks
            except Exception as exc:  # pragma: no cover - network failures
                LOGGER.warning("Azure AI Search query failed: %s", exc)

        lowered = query.lower()
        tokens = lowered.split()
        fallback: List[RetrievalChunk] = []
        for chunk in self._local_index:
            if page_id and not chunk.id.startswith(page_id):
                continue
            content_lower = chunk.content.lower()
            if all(token in content_lower for token in tokens):
                fallback.append(chunk)
        return fallback[:5]

    async def _compute_vector(self, text: str) -> Optional[List[float]]:
        deployment = self.settings.azure_openai_deployment_embeddings
        if not deployment:
            return None
        url = (
            f"{self.settings.azure_openai_endpoint}/openai/deployments/{deployment}/embeddings"
            "?api-version=2024-02-15-preview"
        )
        headers = {
            "api-key": self.settings.azure_openai_api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json={"input": text})
            response.raise_for_status()
            payload = response.json()
        vector = payload.get("data", [{}])[0].get("embedding")
        if isinstance(vector, list):
            return vector  # type: ignore[return-value]
        return None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletion(BaseModel):
    role: str
    content: str


class AzureOpenAIChatClient:
    """Wrapper around Azure OpenAI Chat Completions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_version = "2024-02-15-preview"

    async def generate_reply(
        self, conversation: Iterable[ChatMessage], grounding: Iterable[RetrievalChunk]
    ) -> Tuple[ChatCompletion, List[str]]:
        messages = [{"role": msg.role, "content": msg.content} for msg in conversation]
        if grounding:
            context = "\n\n".join(
                f"Source: {chunk.source}\n{chunk.content}" for chunk in grounding
            )
            messages.append(
                {
                    "role": "system",
                    "content": "Use the following retrieved context to guide your response:\n" + context,
                }
            )
        deployments: List[str] = [self.settings.azure_openai_deployment_gpt4o]
        if self.settings.azure_openai_deployment_gpt4o_mini:
            deployments.append(self.settings.azure_openai_deployment_gpt4o_mini)
        last_error: Optional[Exception] = None
        sources = [chunk.source for chunk in grounding]
        for deployment in deployments:
            try:
                completion = await self._invoke_chat_completion(deployment, messages)
                return completion, sources
            except Exception as exc:  # pragma: no cover - network failures
                LOGGER.warning("Azure OpenAI request failed for %s: %s", deployment, exc)
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("Azure OpenAI call failed with no deployments available")

    async def _invoke_chat_completion(
        self, deployment: str, messages: List[Dict[str, str]]
    ) -> ChatCompletion:
        url = (
            f"{self.settings.azure_openai_endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.settings.azure_openai_api_key,
            "Content-Type": "application/json",
        }
        payload = {"messages": messages, "temperature": 0.2, "top_p": 0.95}
        async for attempt in AsyncRetrying(
            wait=wait_exponential(min=1, max=8),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return ChatCompletion(role=message.get("role", "assistant"), content=message.get("content", ""))


class AzureRedisStore:
    """Redis wrapper storing session state with in-memory fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._memory_store: Dict[str, Any] = {}
        self._redis = None
        try:  # pragma: no cover - optional dependency
            import redis.asyncio as redis_asyncio

            self._redis = redis_asyncio.from_url(
                settings.redis_url, encoding="utf-8", decode_responses=True
            )
        except Exception as exc:  # pragma: no cover - dependency not available
            LOGGER.warning("Falling back to in-memory Redis store: %s", exc)
            self._redis = None

    async def store_page(
        self, thread_id: Optional[str], page: Dict[str, Any], fields: List[Dict[str, Any]]
    ) -> bool:
        payload = json.dumps({"page": page, "fields": fields})
        key = thread_id or f"memory::{page.get('id', 'unknown')}"
        if self._redis is not None and thread_id:
            try:
                await self._redis.set(key, payload, ex=60 * 60)
                return True
            except Exception as exc:  # pragma: no cover - connection issue
                LOGGER.warning("Redis write failed (%s), falling back to memory", exc)
        self._memory_store[key] = payload
        return True


class AzureCosmosLogger:
    """Audit logger writing chat/validation/apply traces to Cosmos DB."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._file_path = Path("backend_cosmos_audit.jsonl")
        self._container_client = self._init_container()

    def _init_container(self):
        if not (CosmosClient and PartitionKey):  # pragma: no cover - optional dependency
            return None
        try:
            client = CosmosClient(self.settings.cosmos_endpoint, credential=self.settings.cosmos_key)  # type: ignore[arg-type]
            database = client.create_database_if_not_exists(self.settings.cosmos_db_name)
            container = database.create_container_if_not_exists(
                id=self.settings.cosmos_container_name,
                partition_key=PartitionKey(path="/session/threadId"),
                offer_throughput=400,
            )
            return container
        except Exception as exc:  # pragma: no cover - network failures
            LOGGER.warning("Failed to initialise Cosmos container: %s", exc)
            return None

    def _write_local(self, record: Dict[str, Any]) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _write_remote(self, record: Dict[str, Any]) -> None:
        if not self._container_client:
            return
        try:
            payload = dict(record)
            payload.setdefault("id", f"{record['type']}-{uuid.uuid4()}")
            self._container_client.upsert_item(payload)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - network failures
            LOGGER.warning("Failed to write record to Cosmos DB: %s", exc)

    def log_chat(
        self,
        session: Dict[str, Any],
        page: Dict[str, Any],
        prompt: Optional[str],
        response: str,
        sources: List[str],
    ) -> None:
        record = {
            "type": "chat",
            "timestamp": datetime.utcnow().isoformat(),
            "session": session,
            "page": page,
            "prompt": prompt,
            "response": response,
            "sources": sources,
        }
        self._write_local(record)
        self._write_remote(record)

    def log_validation(self, session: Dict[str, Any], page: Dict[str, Any], validation: Any) -> None:
        record = {
            "type": "validation",
            "timestamp": datetime.utcnow().isoformat(),
            "session": session,
            "page": page,
            "validation": getattr(validation, "dict", lambda: validation)(),
        }
        self._write_local(record)
        self._write_remote(record)

    def log_apply(
        self,
        session: Dict[str, Any],
        page: Dict[str, Any],
        fields: List[Dict[str, Any]],
        stored: bool,
    ) -> None:
        record = {
            "type": "apply",
            "timestamp": datetime.utcnow().isoformat(),
            "session": session,
            "page": page,
            "fields": fields,
            "stored": stored,
        }
        self._write_local(record)
        self._write_remote(record)
