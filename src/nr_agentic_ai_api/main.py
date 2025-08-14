"""
Main FastAPI application with POST endpoint backbone
"""

import os
import logging
from typing import Optional, List, TypedDict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure at least one handler is configured so logs are emitted under uvicorn
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


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

def _get_search_client() -> Optional[SearchClient]:
    """Get or lazily initialize the Azure AI Search client.

    Uses a function attribute for caching to avoid module-level globals.
    Returns None if not configured.
    """
    if hasattr(_get_search_client, "_client"):
        return getattr(_get_search_client, "_client")  # type: ignore[attr-defined]

    search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    search_key = os.environ.get("AZURE_SEARCH_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")

    client: Optional[SearchClient] = None
    if search_endpoint and search_key and index_name:
        credential = AzureKeyCredential(search_key)
        client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=credential,
        )
    setattr(_get_search_client, "_client", client)  # type: ignore[attr-defined]
    return client


@tool("ai_search_tool")
def ai_search_tool(query: str) -> str:
    """
    Search and retrieve data from the configured Azure AI Search index.

    Expects the following environment variables to be set:
    - AZURE_SEARCH_ENDPOINT
    - AZURE_SEARCH_KEY
    - AZURE_SEARCH_INDEX_NAME
    """
    client = _get_search_client()

    if client is None:
        return (
            "Azure Search not configured. Please set AZURE_SEARCH_ENDPOINT, "
            "AZURE_SEARCH_KEY, and AZURE_SEARCH_INDEX_NAME."
        )

    try:
        search_results = client.search(
            search_text=query,
            select=["*"],
            top=5,
        )
        results = [dict(r) for r in search_results]
        if not results:
            return f"No results found for query '{query}'"
        # Return a concise string representation of results
        return str(results)
    except AzureError as e:
        return f"Error searching index: {str(e)}"


def _init_tavily_client() -> TavilyClient:
    """Initialize the Tavily client with API key and URL."""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError(
            "Tavily API not configured. Please set TAVILY_API_KEY."
        )

    return TavilyClient(tavily_api_key)

@tool("tavily_search_tool")
def tavily_search_tool(query: str) -> str:
    """
    Search and retrieve data from the Tavily API.
    """
    try:
        tavily_client = _init_tavily_client()
        response = tavily_client.search(query)
        # Tavily client returns a dict, not an HTTP response object.
        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return f"No results found for query '{query}'"
        
        # Return a concise string representation of results
        return str(results)
    except Exception as e:
        # Never raise from tools; return a friendly error string so the agent can conclude.
        return f"Tavily search error: {e}"


# Create the Land agent
land_tools = [tavily_search_tool]
land_prompt = PromptTemplate.from_template(
    "You are the Land agent. You can use tools to answer the user's request.\n\n"
    "You have access to the following tools:\n{tools}\n\n"
    "When deciding what to do, follow this format exactly:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, must be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation cycle can repeat) ...\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
)
land_agent = create_react_agent(llm, land_tools, land_prompt)
land_executor = AgentExecutor(
    agent=land_agent,
    tools=land_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)

# Create the Water agent
water_tools = [tavily_search_tool]
water_prompt = PromptTemplate.from_template(
    "You are the Water agent. You can use tools to answer the user's request.\n\n"
    "You have access to the following tools:\n{tools}\n\n"
    "When deciding what to do, follow this format exactly:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, must be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation cycle can repeat) ...\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
)
water_agent = create_react_agent(llm, water_tools, water_prompt)
water_executor = AgentExecutor(
    agent=water_agent,
    tools=water_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)

# Create the orchestrator agent (without tools)
orchestrator_prompt = PromptTemplate.from_template(
    """
You are a routing assistant. Decide which single specialized agent should handle the user's request.

Return exactly one word (lowercase):
- land
- water

User request: {input}
"""
)
# Simple chain (no tools) for routing: prompt -> llm -> string
orchestrator_router = orchestrator_prompt | llm | StrOutputParser()


# Define workflow state
class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow."""
    input: str
    route: str
    response: str


# Define workflow nodes
async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that asks the LLM chain to choose 'land' or 'water'."""
    result = await orchestrator_router.ainvoke({"input": state["input"]})
    # Be defensive: take only the first token
    route = str(result).strip().split()[0].lower()
    if route not in {"land", "water"}:
        route = "land"
    return {"route": route}


async def land_node(state: WorkflowState) -> WorkflowState:
    """Land node that processes the request with tools"""
    result = await land_executor.ainvoke({"input": state["input"]})
    output_text = (
        result.get("output") if isinstance(result, dict) else None
    ) or str(result)
    return {"response": output_text}


async def water_node(state: WorkflowState) -> WorkflowState:
    """Water node that processes the request with tools"""
    result = await water_executor.ainvoke({"input": state["input"]})
    output_text = (
        result.get("output") if isinstance(result, dict) else None
    ) or str(result)
    return {"response": output_text}


# Create the workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("land", land_node)
workflow.add_node("water", water_node)
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    lambda s: s["route"],
    {
        "land": "land",
        "water": "water",
    },
)
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
        # Log the incoming request
        logger.info("Processing request: %s", request.message)
        if request.formFields:
            logger.info("Form fields count: %d", len(request.formFields))
        # Use the LangGraph workflow (async) to process the request
        workflow_result = await app_workflow.ainvoke({"input": request.message})

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
