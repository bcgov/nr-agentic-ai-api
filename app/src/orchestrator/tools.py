from typing import List, Dict
import warnings

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from .settings import SearchSettings

settings = SearchSettings.from_env()


def _search_client() -> SearchClient:
    settings.validate()
    assert settings.endpoint is not None  # for type checkers
    assert settings.key is not None
    return SearchClient(settings.endpoint, settings.index, AzureKeyCredential(settings.key))

def rag_retrieve(query: str, section: str, k: int = 3) -> List[Dict]:
    """Simple retrieval; we only use it to prove RAG wiring (we don't LLM-generate in this MVP)."""
    try:
        client = _search_client()
    except RuntimeError as err:
        warnings.warn(str(err))
        return []
    results = client.search(search_text=query, filter=f"section eq '{section}'", top=k)
    out = []
    for r in results:
        out.append({"url": r.get("url"), "content": r.get("content")})
    return out

def mapping_refs(ej, field_ids: List[str]) -> List[str]:
    refs = []
    for fid in field_ids:
        meta = ej.metadata.get(fid)
        if meta and meta.externalRef:
            refs.append(meta.externalRef)
    return sorted(set(refs))[:3]
