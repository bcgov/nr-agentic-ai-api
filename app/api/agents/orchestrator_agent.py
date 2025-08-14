from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import json
import os
import re
from app.core.logging import get_logger
from app.core.config import (
    settings,
)  # Assuming mapping document is loaded from a config module

MAPPING_DOC = settings.MAPPING_DOC  # Load mapping document from settings

# Initialize structured logger
logger = get_logger(__name__)

# Azure configurations
search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "bc-water-index"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY")),
)
AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-12-01-preview",
)


# Subagent functions
def source_agent(query, _=None):
    results = search_client.search(search_text=query, top=3)
    formatted_results = [
        result["content"] for result in results if result.get("content")
    ]
    return (
        formatted_results
        if formatted_results
        else ["No relevant data found for source query."]
    )


def usage_agent(query, _=None):
    results = search_client.search(search_text=query, top=3)
    formatted_results = [
        result["content"] for result in results if result.get("content")
    ]
    return (
        formatted_results
        if formatted_results
        else ["No relevant data found for usage query."]
    )


def permissions_agent(query, _=None):
    results = search_client.search(
        search_text=f"{query} +BC Water Sustainability Act", top=3
    )
    formatted_results = [
        result["content"] for result in results if result.get("content")
    ]
    return formatted_results if formatted_results else ["No compliance data found."]


# Store results in Blob Storage
def store_result(blob_name, data):
    blob_client = blob_service_client.get_blob_client(
        container="results", blob=blob_name
    )
    blob_client.upload_blob(json.dumps(data), overwrite=True)


# Parse JSON
def parse_json(json_data):
    try:
        if len(json.dumps(json_data)) > 1_000_000:  # 1MB limit
            logger.error("JSON input too large", json_size=len(json.dumps(json_data)))
            return {"error": "JSON input too large"}
        fields = json.loads(json_data)
        return {MAPPING_DOC.get(key, key): value for key, value in fields.items()}
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON format",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return {"error": "Invalid JSON format"}


# Route queries to subagents
def route_query(field, value):
    field_lower = field.lower()
    patterns = {
        "SourceAgent": r"(water_)?source|river|lake|stream",
        "UsageAgent": r"usage|consumption|irrigation|industrial",
        "PermissionsAgent": r"permit|license|compliance|regulation",
    }
    for agent, pattern in patterns.items():
        if re.search(pattern, field_lower):
            logger.info("Routing field to agent", field=field, agent=agent)
            return globals()[agent.lower()](field, value)
    logger.warning("Unknown field encountered", field=field)
    return "Unknown field"


# Process JSON with prompts
def process_json(json_data):
    logger.info("Processing JSON data", json_data=json_data)
    mapped_fields = parse_json(json_data)
    if "error" in mapped_fields:
        return mapped_fields
    results = {}
    missing_fields = []
    prompts = {
        "water_source": "Please specify the water source (e.g., Fraser River, Okanagan Lake).",
        "usage_type": "Please provide the water usage type (e.g., irrigation, industrial).",
        "permit_status": "Please indicate the permit status or requirements.",
    }
    for field, value in mapped_fields.items():
        if not value:
            missing_fields.append(
                {
                    "field": field,
                    "prompt": prompts.get(field, f"Please provide {field}."),
                }
            )
        else:
            results[field] = route_query(field, value)
            store_result(f"result_{field}.json", results[field])
    if missing_fields:
        logger.info(
            "Processing incomplete - missing fields", missing_fields=missing_fields
        )
        return {
            "status": "incomplete",
            "missing_fields": missing_fields,
            "results": results,
        }
    logger.info("JSON processing completed successfully")
    return {
        "status": "complete",
        "results": results,
        "message": "Form completed successfully!",
    }


# Create Orchestrator agent
orchestrator_tools = [
    Tool(
        name="RouteQuery",
        func=route_query,
        description="Routes field queries to subagents",
    ),
    Tool(
        name="SourceAgent", func=source_agent, description="Queries water source data"
    ),
    Tool(name="UsageAgent", func=usage_agent, description="Queries water usage data"),
    Tool(
        name="PermissionsAgent",
        func=permissions_agent,
        description="Queries permit data",
    ),
]
prompt = """You are an Orchestrator for a BC Water License form assistant. Analyze the enriched JSON.
Sections: Source (water sources/works), Usage (purposes/quantities), Perms (authorizations/Crown).
Steps:
1. Identify incomplete/required fields by section.
2. Route to agents: Source first (foundational), then Usage, then Perms.
3. If dependency (e.g., source affects purpose), sequence calls.
4. For clarifications, generate user-friendly questions.
5. Aggregate when all required fields are filled (check mapping metadata).
Output JSON: {"routes": [list of agent calls], "clarifications": [questions], "finalValues": {{...}} if complete}}.
"""

orchestrator = create_react_agent(llm=llm, tools=orchestrator_tools, prompt=prompt)
orchestrator_executor = AgentExecutor(agent=orchestrator, tools=orchestrator_tools)
