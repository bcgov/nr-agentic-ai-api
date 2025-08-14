from math import log
import os
from datetime import datetime
from typing import List, Optional, TypedDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()
load_dotenv()


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version="2024-12-01-preview",
)


"""Azure AI Search tool implementation using @tool decorator."""

# Lazily initialized search client shared by the tool
_search_client: Optional[SearchClient] = None


def _init_search_client() -> None:
    """Initialize the global Azure AI Search client if env vars are present."""
    global _search_client
    if _search_client is not None:
        return

    search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    search_key = os.environ.get("AZURE_SEARCH_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")

    if search_endpoint and search_key and index_name:
        credential = AzureKeyCredential(search_key)
        _search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential,
        )


@tool("ai_search_tool")
def ai_search_tool(query: str) -> str:
    """
    Search and retrieve data from the configured Azure AI Search index.

    Expects the following environment variables to be set:
    - AZURE_SEARCH_ENDPOINT
    - AZURE_SEARCH_KEY
    - AZURE_SEARCH_INDEX_NAME
    """
    _init_search_client()

    if _search_client is None:
        return (
            "Azure Search not configured. Please set AZURE_SEARCH_ENDPOINT, "
            "AZURE_SEARCH_KEY, and AZURE_SEARCH_INDEX_NAME."
        )

    try:
        search_results = _search_client.search(
            search_text=query,
            select=["*"],
            top=5,
        )
        results = [dict(r) for r in search_results]
        if not results:
            return f"No results found for query '{query}'"
        # Return a concise string representation of results
        return str(results)
    except Exception as e:
        return f"Error searching index: {str(e)}"


# Create the Source agent
source_agent_tools = [ai_search_tool]
source_agent_prompt = """
    Prompt: 
        You are a Source Agent for BC Water License. Focus on sources/works fields: {fields}.
        Use RAG to ground in docs like Water Sustainability Act, [404] restrictions.
        Steps:
        1. Validate inputs (e.g., required fields).
        2. Suggest values/explain (e.g., if well, check proximity).
        3. If unclear, ask targeted questions.
        Output JSON: {{"updatedFields": {{...}}, "clarifications": [questions], "references": [doc excerpts]}}.
    """

source_agent = create_react_agent(llm, source_agent_tools, source_agent_prompt)

# Create the Usage agent
usage_agent_tools = [ai_search_tool]
usage_agent_prompt = """
    You are a Usage Agent. Handle purposes/quantities: {fields}.
    Ground in docs: Purpose defs PDF, Unit Converter, Ag Calculator.
    Steps:
    1. Suggest purposes based on user intent.
    2. Calculate/convert (e.g., m³/day from inputs).
    3. Validate seasonality/requirements.
    Output JSON: {{"updatedFields": {{...}}, "clarifications": [...], "references": [...]}}.
    """

usage_agent = create_react_agent(llm, usage_agent_tools, usage_agent_prompt)

# Create the Usage agent
permissions_agent_tools = [ai_search_tool]
permissions_agent_prompt = """
    You are a Permissions Agent. Manage authorizations/Crown: {fields}.
    Ground in docs: Crown Land Uses, Joint Agreement PDF.
    Steps:
    1. Run checklist questions.
    2. Flag requirements (e.g., if Yes, suggest contact).
    3. Explain exemptions.
    Output JSON: {{"updatedFields": {{...}}, "clarifications": [...], "references": [...]}}.
    """

permissions_agent = create_react_agent(
    llm, permissions_agent_tools, permissions_agent_prompt
)

orchestrator_prompt = PromptTemplate.from_template(
    "You are an orchestrator agent. Delegate the user's request to the "
    "Land and Water agents.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\n"
    "Tool names: {tool_names}\n\n"
    "Simply acknowledge that you will delegate this to the Land and Water agents.\n\n"
    "{agent_scratchpad}"
)
orchestrator_agent = create_react_agent(llm, [], orchestrator_prompt)


# Define workflow state
class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow."""

    input: str
    response: str


# Define workflow nodes
async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that delegates to Land and Water agents"""
    result = await orchestrator_agent.ainvoke({"input": state["input"]})
    return {"response": result["messages"][-1].content}


# Create the workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", END)
# workflow.add_node("land", land_node)
# workflow.add_node("water", water_node)
# workflow.add_edge(START, "orchestrator")
# workflow.add_edge("orchestrator", "land")
# workflow.add_edge("orchestrator", "water")
# workflow.add_edge("land", END)
# workflow.add_edge("water", END)

# Compile the workflow
app_workflow = workflow.compile()


# Form field model for the JSON array
class FormField(BaseModel):
    """Model for individual form fields"""

    data_id: str = None
    fieldLabel: str = None
    fieldType: str = None
    fieldValue: str = None


# Base request model for POST endpoint
class RequestModel(BaseModel):
    """Base request model for the POST endpoint"""

    message: str
    formFields: Optional[List[FormField]] = None


# Base response model
class ResponseModel(BaseModel):
    """Base response model for the POST endpoint"""

    status: str
    message: str


@router.post("/process", response_model=ResponseModel)
async def process_request(request: RequestModel):
    """
    Main POST endpoint to receive and process requests

    This endpoint uses the orchestrator agent to process incoming requests.
    """
    try:
        logger.info(
            "Processing orchestrator request",
            request_message=request.message,
            form_fields=request.formFields,
            form_fields_count=len(request.formFields) if request.formFields else 0,
        )
        # Use the LangGraph workflow to process the request
        # workflow_result = app_workflow.invoke({"input": request.message})

        return ResponseModel(
            status="success",
            message="success",
        )

    except Exception as e:
        logger.error(
            "Error processing orchestrator request",
            error=str(e),
            error_type=type(e).__name__,
            request_message=request.message,
            form_fields_count=len(request.formFields) if request.formFields else 0,
            has_data=bool(request.data),
            has_metadata=bool(request.metadata),
            timestamp=datetime.now().isoformat(),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
