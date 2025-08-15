"""
Shared LLM infrastructure for hybrid agent processing
"""

import os
import json
import re

from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.core.logging import get_logger

logger = get_logger(__name__)


# Initialize LLM with proper validation
def _initialize_llm():
    """Initialize Azure OpenAI LLM with validation"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    if not endpoint:
        logger.error("AZURE_OPENAI_ENDPOINT not configured")
        raise ValueError("AZURE_OPENAI_ENDPOINT is required")

    if not api_key:
        logger.error("AZURE_OPENAI_API_KEY not configured")
        raise ValueError("AZURE_OPENAI_API_KEY is required")

    logger.info(
        f"Initializing LLM with endpoint: {endpoint[:30]}..., deployment: {deployment}"
    )

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment,
        api_version=api_version,
        temperature=0.3,  # Lower temperature for more consistent reasoning
    )


# Initialize shared LLM instance
try:
    llm = _initialize_llm()
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    llm = None


async def enhanced_analysis(
    agent_type: str,
    query: str,
    initial_analysis: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    form_fields: Optional[List] = None,
) -> Dict[str, Any]:
    """
    Perform LLM-enhanced analysis when rule-based processing needs augmentation
    """
    try:
        # Check if LLM is available
        if llm is None:
            logger.error("LLM not initialized - falling back to rule-based analysis")
            return {
                "error": "LLM not available",
                "fallback": "Using rule-based analysis only",
            }

        # Agent-specific prompts
        prompts = {
            "source": """You are a BC Water License Source Analysis expert. 
Analyze the user query and initial findings to provide enhanced water source recommendations.

User Query: {query}
Initial Analysis: {initial_analysis}
Search Results: {search_results}
Form Fields: {form_fields}

Provide enhanced analysis in JSON format:
{{
    "enhanced_sources": [{{
        "source_type": "river|lake|well|groundwater|reservoir|stream",
        "location": "specific location if mentioned",
        "confidence": 0.0-1.0,
        "reasoning": "why this source is recommended"
    }}],
    "location_insights": ["specific insights about locations mentioned"],
    "regulatory_considerations": ["water rights, seasonal restrictions, etc."],
    "recommendations": ["actionable next steps"],
    "risk_factors": ["potential challenges or limitations"]
}}""",
            "usage": """You are a BC Water License Usage Analysis expert.
Analyze the user query and initial findings to provide enhanced water usage recommendations.

User Query: {query}
Initial Analysis: {initial_analysis}  
Search Results: {search_results}
Form Fields: {form_fields}

Provide enhanced analysis in JSON format:
{{
    "enhanced_usage": [{{
        "purpose": "irrigation|industrial|domestic|mining|power|conservation",
        "estimated_quantity": "volume estimate with units",
        "seasonal_pattern": "usage pattern description",
        "efficiency_rating": "high|medium|low",
        "confidence": 0.0-1.0,
        "reasoning": "why this usage classification"
    }}],
    "quantity_calculations": [{{
        "usage_type": "purpose",
        "annual_estimate": "volume with reasoning",
        "peak_demand": "seasonal peak information"
    }}],
    "efficiency_recommendations": ["water conservation measures"],
    "compliance_requirements": ["usage-specific regulations"],
    "monitoring_needs": ["measurement and reporting requirements"]
}}""",
            "permissions": """You are a BC Water License Permissions & Compliance expert.
Analyze the user query and initial findings to provide enhanced regulatory guidance.

User Query: {query}
Initial Analysis: {initial_analysis}
Search Results: {search_results}
Form Fields: {form_fields}

Provide enhanced analysis in JSON format:
{{
    "enhanced_requirements": [{{
        "requirement_type": "license|permit|assessment|consultation",
        "description": "specific requirement description", 
        "priority": "high|medium|low",
        "timeline": "estimated time to complete",
        "confidence": 0.0-1.0,
        "reasoning": "regulatory basis"
    }}],
    "fee_exemption_analysis": {{
        "eligible": true|false,
        "category": "government|first_nation|other",
        "supporting_evidence": ["evidence for eligibility"],
        "required_documentation": ["documents needed"]
    }},
    "consultation_requirements": [{{
        "type": "First Nation|Environmental|Public", 
        "description": "consultation details",
        "timeline": "estimated duration"
    }}],
    "compliance_checklist": ["step-by-step compliance actions"],
    "risk_assessment": ["regulatory risks and mitigation strategies"]
}}""",
        }

        if agent_type not in prompts:
            logger.warning(f"Unknown agent type: {agent_type}")
            return {"error": f"Unknown agent type: {agent_type}"}

        # Format the prompt
        prompt_template = prompts[agent_type]
        formatted_prompt = prompt_template.format(
            query=query,
            initial_analysis=str(initial_analysis),
            search_results=str(search_results[:3]),  # Limit to top 3 results
            form_fields=str(form_fields) if form_fields else "None",
        )

        # Create messages
        messages = [
            SystemMessage(
                content="You are an expert in BC water licensing and regulations. Always respond with valid JSON."
            ),
            HumanMessage(content=formatted_prompt),
        ]

        # Get LLM response
        response = await llm.ainvoke(messages)

        response_text = response.content

        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            enhanced_data = json.loads(json_match.group())
        else:
            # Fallback if no JSON found
            enhanced_data = {
                "error": "Could not parse LLM response as JSON",
                "raw_response": response_text,
            }

        logger.info(f"Enhanced analysis completed for {agent_type}")
        return enhanced_data

    except Exception as e:
        logger.error(
            f"Error in enhanced analysis for {agent_type}: {str(e)}", exc_info=True
        )
        return {
            "error": f"Enhanced analysis failed: {str(e)}",
            "fallback": "Using rule-based analysis only",
        }


def needs_enhancement(initial_analysis: Dict[str, Any], query: str) -> bool:
    """
    Determine if the initial analysis needs LLM enhancement
    Only return True if LLM is available and rule-based analysis is insufficient
    """
    # Don't attempt enhancement if LLM is not available
    if llm is None:
        logger.debug("LLM not available - skipping enhancement check")
        return False

    # Check if rule-based analysis found limited results
    if (
        not initial_analysis.get("detected_sources")
        and not initial_analysis.get("detected_usage")
        and not initial_analysis.get("detected_requirements")
    ):
        return True

    # Check for ambiguous or complex language in query
    complex_indicators = [
        "approximately",
        "around",
        "similar to",
        "like",
        "near",
        "complex",
        "multiple",
        "various",
        "depends on",
        "varies",
        "not sure",
        "unclear",
        "might need",
        "could be",
    ]

    query_lower = query.lower()
    if any(indicator in query_lower for indicator in complex_indicators):
        return True

    # Check if query length suggests complexity
    if len(query.split()) > 15:
        return True

    # Check for multiple concepts in one query
    concept_count = 0
    concepts = ["source", "usage", "permit", "license", "exemption", "compliance"]
    for concept in concepts:
        if concept in query_lower:
            concept_count += 1

    if concept_count > 2:
        return True

    return False
