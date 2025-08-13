import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()
load_dotenv()


# Initialize the Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version="2024-12-01-preview",
)


# Create a tool for reading data from AI search index
class AISearchTool(BaseTool):
    """Tool for searching and retrieving data from AI Search index."""

    name: str = "ai_search_tool"
    description: str = "A tool that searches and retrieves data from AI Search index"
    search_client: Optional[SearchClient] = None

    def __init__(self):
        super().__init__()

        # Get environment variables
        search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        search_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        index_name = "bc-water-index"  # currently hardcoded , see web_crawler.py

        # Initialize search client if environment variables are set
        if search_endpoint and search_key and index_name:
            credential = AzureKeyCredential(search_key)
            self.search_client = SearchClient(
                endpoint=search_endpoint, index_name=index_name, credential=credential
            )

    def _run(self, query: str, run_manager=None) -> str:
        if not self.search_client:
            return "Azure Search not configured."

        try:
            # Search the index
            search_results = self.search_client.search(
                search_text=query, select=["*"], top=5
            )

            # Process results
            results = []
            for result in search_results:
                results.append(dict(result))

            if not results:
                return f"No results found for query '{query}'"

        except Exception as e:
            return f"Error searching index: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# Create the Land agent
land_tools = [AISearchTool()]
land_prompt = PromptTemplate.from_template(
    "You are a Land agent. Use the available tools to process the "
    "user's request.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n{agent_scratchpad}"
)

land_agent = create_react_agent(llm, land_tools, land_prompt)
land_agent_executor = AgentExecutor(agent=land_agent, tools=land_tools, verbose=True)

# Create the Water agent
water_tools = [AISearchTool()]
water_prompt = PromptTemplate.from_template(
    "You are a Water agent. Use the available tools to process the "
    "user's request.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n{agent_scratchpad}"
)

water_agent = create_react_agent(llm, water_tools, water_prompt)
water_agent_executor = AgentExecutor(agent=water_agent, tools=water_tools, verbose=True)

# Create the orchestrator agent (without tools)
orchestrator_prompt = PromptTemplate.from_template(
    "You are an orchestrator agent. Delegate the user's request to the "
    "Land and Water agents.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n"
    "Simply acknowledge that you will delegate this to the Land and Water agents.\n\n"
    "{agent_scratchpad}"
)

orchestrator_agent = create_react_agent(llm, [], orchestrator_prompt)
orchestrator_executor = AgentExecutor(agent=orchestrator_agent, tools=[], verbose=True)


# Define workflow state
class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow."""

    input: str
    orchestrator_output: str
    land_output: str
    water_output: str


# Define workflow nodes
def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that delegates to Land and Water agents"""
    result = orchestrator_executor.invoke({"input": state["input"]})
    return {"orchestrator_output": result["output"]}


def land_node(state: WorkflowState) -> WorkflowState:
    """Land node that processes the request with tools"""
    result = land_agent_executor.invoke({"input": state["input"]})
    return {"land_output": result["output"]}


def water_node(state: WorkflowState) -> WorkflowState:
    """Water node that processes the request with tools"""
    result = water_agent_executor.invoke({"input": state["input"]})
    return {"water_output": result["output"]}


# Create the workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("land", land_node)
workflow.add_node("water", water_node)
workflow.set_entry_point("orchestrator")
workflow.add_edge("orchestrator", "land")
workflow.add_edge("orchestrator", "water")
workflow.add_edge("land", END)
workflow.add_edge("water", END)

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
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# Base response model


class ResponseModel(BaseModel):
    """Base response model for the POST endpoint"""

    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str


router = APIRouter()


@router.post("/process", response_model=ResponseModel)
async def process_request(request: RequestModel):
    """
    Main POST endpoint to receive and process requests

    This endpoint uses the orchestrator agent to process incoming requests.
    """
    try:
        logger.info("Processing request", request=request)

        # Use the LangGraph workflow to process the request
        workflow_result = app_workflow.invoke({"input": request.message})
        logger.info("Workflow result", result=workflow_result)
        # Process the request with workflow results
        processed_data = {
            "received_message": request.message,
            "orchestrator_output": workflow_result["orchestrator_output"],
            "land_output": workflow_result["land_output"],
            "water_output": workflow_result["water_output"],
            "received_form_fields": request.formFields,
            "received_data": request.data,
            "received_metadata": request.metadata,
            "processed_at": datetime.now().isoformat(),
        }

        return ResponseModel(
            status="success",
            message="Request processed successfully by orchestrator agent",
            data=processed_data,
            timestamp=datetime.now().isoformat(),
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
            status_code=500, detail=(f"Error processing request: {str(e)}")
        )
