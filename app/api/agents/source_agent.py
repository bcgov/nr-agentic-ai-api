import json
import os
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from app.core.logging import get_logger
from .llm_helper import enhanced_analysis, needs_enhancement

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Azure Cognitive Search (READS use the QUERY key; keep vars consistent)
# ------------------------------------------------------------------------------
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv(
    "AZURE_SEARCH_INDEX_NAME", "bc-water-index"
)  # single source of truth across repo
SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")


_search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_QUERY_KEY),
)


# ------------------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------------------
def _search(
    query: str,
    *,
    top: int = 3,
    select: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the index and return a compact list of dicts suitable for downstream use.
    """
    if not select:
        # keep projection tight; adjust to your index schema
        select = ["id", "title", "url", "content"]

    try:
        results = _search_client.search(search_text=query, select=["*"], top=top)
        docs: List[Dict[str, Any]] = []
        for r in results:
            # `r` behaves like a dict; include score if present
            doc = {k: r.get(k) for k in select if k in r}
            score = r.get("@search.score")
            if score is not None:
                doc["score"] = score
            # provide a short snippet if content is long
            content = doc.get("content")
            if isinstance(content, str) and len(content) > 400:
                doc["snippet"] = content[:400] + "…"
            docs.append(doc)
        return docs
    except Exception as e:
        logger.exception("SourceAgent search failed: %s", e)
        return []


# ------------------------------------------------------------------------------
# Public API expected by the orchestrator
#   - source_agent(query) -> JSON string (for LangChain Tool safety)
#   - invoke_source_agent(query) -> dict (async-friendly wrapper)
# ------------------------------------------------------------------------------
def source_agent(query: str, *_args, **_kwargs) -> str:
    """
    Tool-safe entrypoint: accepts a single string and returns a JSON string.
    Payload schema:
      {
        "agent": "SourceAgent",
        "query": "<query>",
        "documents": [ {id,title,url,snippet?,content?,score?}, ... ],
        "message": "<empty or explanation>"
      }
    """
    docs = _search(query, top=3)
    payload = {
        "agent": "SourceAgent",
        "query": query,
        "documents": docs,
        "message": "" if docs else "No relevant data found for source query.",
    }
    return json.dumps(payload, default=str)


async def invoke_source_agent(
    query: str, form_fields: Optional[List] = None
) -> Dict[str, Any]:
    """
    Async-friendly wrapper that returns a dict with comprehensive source analysis.
    Now accepts form_fields for enhanced analysis.
    """
    try:
        logger.info(
            f"Processing source query: {query}",
            extra={"form_fields_count": len(form_fields) if form_fields else 0},
        )

        # The underlying Azure SDK call is synchronous; we call it in-place.
        docs = _search(query, top=5)

        # Extract source types and locations from query
        source_keywords = {
            "fraser river": {
                "type": "river",
                "description": "Major river system in British Columbia",
            },
            "lake": {
                "type": "surface_water",
                "description": "Natural or artificial lake water source",
            },
            "well": {
                "type": "groundwater",
                "description": "Groundwater extraction point",
            },
            "creek": {
                "type": "surface_water",
                "description": "Small watercourse or stream",
            },
            "groundwater": {
                "type": "groundwater",
                "description": "Underground water source",
            },
            "reservoir": {
                "type": "surface_water",
                "description": "Water storage facility",
            },
            "stream": {
                "type": "surface_water",
                "description": "Natural flowing watercourse",
            },
        }

        query_lower = query.lower()
        detected_sources = []

        for keyword, info in source_keywords.items():
            if keyword in query_lower:
                detected_sources.append(
                    {
                        "keyword": keyword,
                        "type": info["type"],
                        "description": info["description"],
                    }
                )

        # Analyze form fields for source information
        source_fields = []
        if form_fields:
            for field in form_fields:
                if hasattr(field, "fieldLabel") and field.fieldLabel:
                    if any(
                        term in field.fieldLabel.lower()
                        for term in ["source", "location", "body of water", "intake"]
                    ):
                        source_fields.append(
                            {
                                "data_id": getattr(field, "data_id", None),
                                "label": field.fieldLabel,
                                "type": getattr(field, "fieldType", None),
                                "value": getattr(field, "fieldValue", None),
                            }
                        )

        # Initial rule-based analysis
        initial_analysis = {
            "detected_sources": detected_sources,
            "source_fields": source_fields,
            "search_results": docs,
            "analysis": f"Found {len(detected_sources)} potential water sources and {len(docs)} relevant documents",
        }

        # Determine if LLM enhancement is needed
        needs_llm_enhancement = needs_enhancement(initial_analysis, query)
        enhanced_data = {}

        if needs_llm_enhancement:
            logger.info("Source analysis requires LLM enhancement")
            enhanced_data = await enhanced_analysis(
                "source", query, initial_analysis, docs, form_fields
            )
        else:
            logger.info("Rule-based source analysis sufficient")

        return {
            "agent": "SourceAgent",
            "status": "success",
            "query": query,
            "documents": docs,
            "detected_sources": detected_sources,
            "source_fields": source_fields,
            "analysis": initial_analysis["analysis"],
            "enhanced_analysis": enhanced_data if needs_llm_enhancement else None,
            "processing_method": "hybrid_llm"
            if needs_llm_enhancement
            else "rule_based",
            "recommendations": enhanced_data.get(
                "recommendations",
                [
                    "Specify exact coordinates if using groundwater",
                    "Provide water rights documentation for surface water",
                    "Include seasonal flow information for streams/rivers",
                ],
            )
            if detected_sources
            else ["Please specify the water source location and type"],
            "message": "" if docs else "No relevant data found for source query.",
        }

    except Exception as e:
        logger.error(f"Error in invoke_source_agent: {str(e)}", exc_info=True)
        return {
            "agent": "SourceAgent",
            "status": "error",
            "query": query,
            "error": str(e),
            "documents": [],
            "detected_sources": [],
            "source_fields": [],
            "analysis": f"Error processing source request: {str(e)}",
            "message": f"Error: {str(e)}",
        }
