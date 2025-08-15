from __future__ import annotations

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
# Internal search helper (compact projection + optional snippet)
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
        # Keep projection tight; adjust fields to your index schema.
        select = ["*", "title", "url", "content"]

    try:
        results = _search_client.search(search_text=query, select=["*"], top=top)
        docs: List[Dict[str, Any]] = []
        for r in results:
            doc = {k: r.get(k) for k in select if k in r}
            score = r.get("@search.score")
            if score is not None:
                doc["score"] = score
            # Provide a short snippet if content is long (non-destructive).
            content = doc.get("content")
            if isinstance(content, str) and len(content) > 400:
                doc["snippet"] = content[:400] + "…"
            docs.append(doc)
        return docs
    except Exception as e:
        logger.exception("UsageAgent search failed: %s", e)
        return []


# ------------------------------------------------------------------------------
# Public API expected by the orchestrator
#   - usage_agent(query) -> JSON string (tool-safe)
#   - invoke_usage_agent(query) -> dict (async-friendly structured)
# Schema mirrors source_agent: {agent, query, documents, message}
# ------------------------------------------------------------------------------
def usage_agent(query: str, *_args, **_kwargs) -> str:
    """
    Tool-safe entrypoint: accepts a single string and returns a JSON string.
    Payload schema:
      {
        "agent": "UsageAgent",
        "query": "<query>",
        "documents": [ {id,title,url,snippet?,content?,score?}, ... ],
        "message": "<empty or explanation>"
      }
    """
    docs = _search(query, top=3)
    payload = {
        "agent": "UsageAgent",
        "query": query,
        "documents": docs,
        "message": "" if docs else "No relevant data found for usage query.",
    }
    return json.dumps(payload, default=str)


async def invoke_usage_agent(
    query: str, form_fields: Optional[List] = None
) -> Dict[str, Any]:
    """
    Async-friendly wrapper that returns a dict with comprehensive usage analysis.
    Now accepts form_fields for enhanced analysis.
    """
    try:
        logger.info(
            f"Processing usage query: {query}",
            extra={"form_fields_count": len(form_fields) if form_fields else 0},
        )

        # The underlying Azure SDK call is synchronous; call directly.
        docs = _search(query, top=5)

        # Extract usage types from query
        usage_keywords = {
            "irrigation": {
                "category": "agricultural",
                "description": "Agricultural irrigation purposes",
                "priority": "high",
            },
            "industrial": {
                "category": "industrial",
                "description": "Industrial processing and manufacturing",
                "priority": "high",
            },
            "domestic": {
                "category": "municipal",
                "description": "Domestic water supply",
                "priority": "medium",
            },
            "mining": {
                "category": "industrial",
                "description": "Mining operations and extraction",
                "priority": "high",
            },
            "power": {
                "category": "industrial",
                "description": "Power generation and hydroelectric use",
                "priority": "high",
            },
            "conservation": {
                "category": "environmental",
                "description": "Water conservation and storage",
                "priority": "medium",
            },
            "cooling": {
                "category": "industrial",
                "description": "Industrial cooling processes",
                "priority": "medium",
            },
            "livestock": {
                "category": "agricultural",
                "description": "Livestock watering",
                "priority": "medium",
            },
            "fire protection": {
                "category": "emergency",
                "description": "Fire protection and suppression",
                "priority": "high",
            },
        }

        query_lower = query.lower()
        detected_usage = []

        for keyword, info in usage_keywords.items():
            if keyword in query_lower:
                detected_usage.append(
                    {
                        "keyword": keyword,
                        "category": info["category"],
                        "description": info["description"],
                        "priority": info["priority"],
                    }
                )

        # Analyze form fields for usage information
        usage_fields = []
        if form_fields:
            for field in form_fields:
                if hasattr(field, "fieldLabel") and field.fieldLabel:
                    if any(
                        term in field.fieldLabel.lower()
                        for term in ["purpose", "use", "usage", "application"]
                    ):
                        usage_fields.append(
                            {
                                "data_id": getattr(field, "data_id", None),
                                "label": field.fieldLabel,
                                "type": getattr(field, "fieldType", None),
                                "value": getattr(field, "fieldValue", None),
                            }
                        )

        # Calculate water quantity estimates if possible
        quantity_estimates = []
        for usage in detected_usage:
            if usage["keyword"] == "irrigation":
                quantity_estimates.append(
                    {
                        "usage": "irrigation",
                        "estimate": "2-5 acre-feet per acre annually",
                    }
                )
            elif usage["keyword"] == "domestic":
                quantity_estimates.append(
                    {
                        "usage": "domestic",
                        "estimate": "0.5-1 acre-foot per household annually",
                    }
                )
            elif usage["keyword"] == "livestock":
                quantity_estimates.append(
                    {
                        "usage": "livestock",
                        "estimate": "Variable based on animal type and count",
                    }
                )

        # Initial rule-based analysis
        initial_analysis = {
            "detected_usage": detected_usage,
            "usage_fields": usage_fields,
            "quantity_estimates": quantity_estimates,
            "search_results": docs,
            "analysis": f"Found {len(detected_usage)} usage types and {len(docs)} relevant documents",
        }

        # Determine if LLM enhancement is needed
        needs_llm_enhancement = needs_enhancement(initial_analysis, query)
        enhanced_data = {}

        if needs_llm_enhancement:
            logger.info("Usage analysis requires LLM enhancement")
            enhanced_data = await enhanced_analysis(
                "usage", query, initial_analysis, docs, form_fields
            )
        else:
            logger.info("Rule-based usage analysis sufficient")

        return {
            "agent": "UsageAgent",
            "status": "success",
            "query": query,
            "documents": docs,
            "detected_usage": detected_usage,
            "usage_fields": usage_fields,
            "quantity_estimates": quantity_estimates,
            "analysis": initial_analysis["analysis"],
            "enhanced_analysis": enhanced_data if needs_llm_enhancement else None,
            "processing_method": "hybrid_llm"
            if needs_llm_enhancement
            else "rule_based",
            "recommendations": enhanced_data.get(
                "efficiency_recommendations",
                [
                    "Specify exact water quantities needed",
                    "Provide detailed usage schedule (seasonal, daily)",
                    "Include efficiency measures planned",
                ],
            )
            if detected_usage
            else ["Please specify the intended water usage purpose"],
            "message": "" if docs else "No relevant data found for usage query.",
        }

    except Exception as e:
        logger.error(f"Error in invoke_usage_agent: {str(e)}", exc_info=True)
        return {
            "agent": "UsageAgent",
            "status": "error",
            "query": query,
            "error": str(e),
            "documents": [],
            "detected_usage": [],
            "usage_fields": [],
            "quantity_estimates": [],
            "analysis": f"Error processing usage request: {str(e)}",
            "message": f"Error: {str(e)}",
        }
