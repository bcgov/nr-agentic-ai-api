"""
Main FastAPI application with POST endpoint backbone
"""

import os
from typing import Optional, List, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END, START
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="NR Agentic AI API",
    description=(
        "An agentic AI API built with FastAPI, LangGraph, and LangChain"
    ),
    version="0.1.0"
)

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


# Create the Land agent
land_tools = [ai_search_tool]
land_prompt = PromptTemplate.from_template(
    "You are a Land agent. Use the available tools to process the "
    "user's request.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n{agent_scratchpad}"
)
land_agent = create_react_agent(llm, land_tools, land_prompt)

# Create the Water agent
water_tools = [ai_search_tool]
water_prompt = PromptTemplate.from_template(
    "You are a Water agent. Use the available tools to process the "
    "user's request.\n\nUser request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n{agent_scratchpad}"
)
water_agent = create_react_agent(llm, water_tools, water_prompt)

# Create the orchestrator agent (without tools)
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


async def land_node(state: WorkflowState) -> WorkflowState:
    """Land node that processes the request with tools"""
    result = await land_agent.ainvoke({"input": state["input"]})
    return {"response": result["messages"][-1].content}


async def water_node(state: WorkflowState) -> WorkflowState:
    """Water node that processes the request with tools"""
    result = await water_agent.ainvoke({"input": state["input"]})
    return {"response": result["messages"][-1].content}


# Create the workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("land", land_node)
workflow.add_node("water", water_node)
workflow.add_edge(START, "orchestrator")
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

# Base response model
class ResponseModel(BaseModel):
    """Base response model for the POST endpoint"""
    status: str
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "NR Agentic AI API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "NR Agentic AI API"}


@app.post("/api/process", response_model=ResponseModel)
async def process_request(request: RequestModel):
    """
    Main POST endpoint to receive and process requests
    
    This endpoint uses the orchestrator agent to process incoming requests.
    """
    try:
        # Use the LangGraph workflow to process the request
        workflow_result = app_workflow.invoke({"input": request.message})
        
        return ResponseModel(
            status="success",
            message=workflow_result["response"],
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        ) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
