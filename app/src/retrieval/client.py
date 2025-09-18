from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "wlrs-index")


def _client() -> Optional[SearchClient]:
    """Create a SearchClient if configured; else None (RAG gracefully no-ops)."""
    if not SEARCH_ENDPOINT or not SEARCH_KEY:
        return None
    return SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))


def _fallback_response(query: str, reason: str) -> Dict[str, Any]:
    return {
        "query": query,
        "snippets": [{"text": "", "source": "", "confidence": 0.0}],
        "fallback": True,
        "error": reason,
    }


def retrieve(query: str, scope: Optional[str] = None, k: int = 3) -> Dict:
    """
    Retrieval contract expected by agents/rules:
    Returns { "query": str, "snippets": [ {text, source, confidence}, ... ] }
    """
    cli = _client()
    if cli is None:
        return _fallback_response(query, "search client not configured")

    flt = f"section eq '{scope}'" if scope else None
    try:
        results = cli.search(search_text=query, filter=flt, top=k)
        records = list(results)
    except Exception as exc:
        # Network restrictions or misconfiguration should fail softly.
        return _fallback_response(query, f"retrieval error: {exc}")

    snippets: List[Dict] = []
    for r in records:
        # Map common field names produced by your indexer
        text = r.get("content") or r.get("chunk") or r.get("text") or ""
        source = r.get("url") or r.get("source") or r.get("path") or ""
        conf = r.get("@search.score") or 0.0
        snippets.append({"text": text, "source": source, "confidence": float(conf)})

    if not snippets:
        return _fallback_response(query, "no search results")

    return {"query": query, "snippets": snippets, "fallback": False, "error": None}
