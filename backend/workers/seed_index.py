"""Seed Azure AI Search with schema metadata and guidance content."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "app" / "schemas"
GUIDANCE_SOURCES = [
    "https://portalext.nrs.gov.bc.ca/web/client/-/water-licence-application.html",
    "https://www2.gov.bc.ca/assets/gov/environment/air-land-water/water/water-rights/water-licensing-forms-fees/water_use_purposes_and_categories.pdf",
]
SEARCH_API_VERSION = "2023-11-01"
EMBEDDING_API_VERSION = "2024-02-15-preview"


@dataclass
class EmbeddingClient:
    endpoint: Optional[str]
    key: Optional[str]
    deployment: Optional[str]

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS"),
        )

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.key and self.deployment)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.available:
            return None
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings"
            f"?api-version={EMBEDDING_API_VERSION}"
        )
        headers = {"api-key": self.key or "", "Content-Type": "application/json"}
        response = httpx.post(url, headers=headers, json={"input": text}, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("data", [{}])[0].get("embedding")
        if isinstance(embedding, list):
            return embedding
        return None


def iter_schema_documents() -> Iterable[Dict[str, object]]:
    for path in SCHEMA_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        page_id = payload.get("id") or path.stem
        yield {
            "id": f"schema::{page_id}",
            "type": "schema",
            "content": json.dumps(payload),
            "source": str(path.relative_to(ROOT)),
            "pageId": page_id,
        }


def fetch_guidance(url: str) -> str:
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    text = response.text
    if len(text) > 4000:
        text = text[:4000]
    return text


def iter_guidance_documents() -> Iterable[Dict[str, object]]:
    for url in GUIDANCE_SOURCES:
        try:
            content = fetch_guidance(url)
        except Exception:
            content = f"Guidance reference: {url}"
        yield {
            "id": f"guidance::{url}",
            "type": "guidance",
            "content": content,
            "source": url,
            "pageId": None,
        }


def ensure_index(endpoint: str, api_key: str, index_name: str, vector_dimensions: Optional[int]) -> None:
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    fields: List[Dict[str, object]] = [
        {"name": "id", "type": "Edm.String", "key": True, "searchable": False},
        {"name": "content", "type": "Edm.String", "searchable": True, "analyzerName": "en.lucene"},
        {"name": "source", "type": "Edm.String", "searchable": True, "filterable": True},
        {"name": "type", "type": "Edm.String", "searchable": False, "filterable": True},
        {"name": "pageId", "type": "Edm.String", "searchable": False, "filterable": True},
    ]
    vector_search = None
    if vector_dimensions:
        fields.append(
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": False,
                "vectorSearchDimensions": vector_dimensions,
                "vectorSearchConfiguration": "openai-vector-config",
            }
        )
        vector_search = {
            "profiles": [
                {
                    "name": "openai-vector-profile",
                    "algorithmConfiguration": "openai-vector-config",
                }
            ],
            "algorithms": [
                {
                    "name": "openai-vector-config",
                    "kind": "hnsw",
                    "parameters": {"m": 4, "efConstruction": 400, "metric": "cosine"},
                }
            ],
        }

    body: Dict[str, object] = {"name": index_name, "fields": fields}
    if vector_search:
        body["vectorSearch"] = vector_search

    url = f"{endpoint}/indexes/{index_name}?api-version={SEARCH_API_VERSION}"
    response = httpx.put(url, headers=headers, json=body, timeout=30.0)
    if response.status_code not in (200, 201):
        response.raise_for_status()


def delete_index(endpoint: str, api_key: str, index_name: str) -> None:
    headers = {"api-key": api_key}
    url = f"{endpoint}/indexes/{index_name}?api-version={SEARCH_API_VERSION}"
    response = httpx.delete(url, headers=headers, timeout=30.0)
    if response.status_code not in (200, 204, 404):
        response.raise_for_status()


def upload_documents(endpoint: str, api_key: str, index_name: str, documents: List[Dict[str, object]]) -> None:
    if not documents:
        return
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    url = f"{endpoint}/indexes/{index_name}/docs/index?api-version={SEARCH_API_VERSION}"
    payload = {"value": documents}
    response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()


def build_documents() -> List[Dict[str, object]]:
    documents = list(iter_schema_documents())
    documents.extend(iter_guidance_documents())
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Azure AI Search index")
    parser.add_argument("--recreate", action="store_true", help="Recreate the index before uploading")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads instead of uploading")
    args = parser.parse_args()

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX", "agentic-water-permit")

    documents = build_documents()

    if args.dry_run or not endpoint or not api_key:
        print("Azure AI Search credentials missing or dry-run requested. Payload preview:")
        print(json.dumps(documents, indent=2))
        return

    embedder = EmbeddingClient.from_env()
    vectors: List[Dict[str, object]] = []
    for document in documents:
        vector = embedder.embed(document["content"]) if embedder.available else None
        payload: Dict[str, object] = {
            "@search.action": "mergeOrUpload",
            "id": document["id"],
            "content": document["content"],
            "source": document["source"],
            "type": document["type"],
            "pageId": document.get("pageId"),
        }
        if vector is not None:
            payload["contentVector"] = vector
        vectors.append(payload)

    if args.recreate:
        delete_index(endpoint, api_key, index_name)

    dimension = len(vectors[0].get("contentVector", [])) if vectors and "contentVector" in vectors[0] else None
    ensure_index(endpoint, api_key, index_name, dimension)
    upload_documents(endpoint, api_key, index_name, vectors)
    print(f"Uploaded {len(vectors)} documents to index '{index_name}'.")


if __name__ == "__main__":
    main()
