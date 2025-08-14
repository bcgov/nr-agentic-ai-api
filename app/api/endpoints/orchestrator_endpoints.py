from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from app.core.logging import get_logger
import json

# Imports for agent invoke functions (assuming module names; adjust based on actual file structure)
from app.api.agents.source_agent import invoke_source_agent
from app.api.agents.permissions_agent import invoke_permissions_agent
from app.api.agents.usage_agent import invoke_usage_agent
from app.api.agents.orchestrator_agent import orchestrator_executor

logger = get_logger(__name__)
load_dotenv()


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
    data: Any = None
    timestamp: str


# Define workflow state
class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow."""

    input: str
    form_fields: Optional[List[FormField]]
    orchestrator_output: str
    source_output: str
    permissions_output: str
    usage_output: str


# Define workflow nodes (async for compatibility)
async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that decides delegation"""
    try:
        result = await orchestrator_executor.ainvoke({"input": state["input"]})
        return {"orchestrator_output": result["output"]}
    except Exception as e:
        logger.error(f"Error in orchestrator node: {str(e)}")
        return {"orchestrator_output": f"Error: {str(e)}"}


async def source_node(state: WorkflowState) -> WorkflowState:
    """Source node that processes the request"""
    try:
        result = await invoke_source_agent(state["input"], state["form_fields"])
        return {"source_output": result}
    except Exception as e:
        logger.error(f"Error in source node: {str(e)}")
        return {"source_output": f"Error: {str(e)}"}


async def permissions_node(state: WorkflowState) -> WorkflowState:
    """Permissions node that processes the request"""
    try:
        result = await invoke_permissions_agent(state["input"], state["form_fields"])
        return {"permissions_output": result}
    except Exception as e:
        logger.error(f"Error in permissions node: {str(e)}")
        return {"permissions_output": f"Error: {str(e)}"}


async def usage_node(state: WorkflowState) -> WorkflowState:
    """Usage node that processes the request"""
    try:
        result = await invoke_usage_agent(state["input"], state["form_fields"])
        return {"usage_output": result}
    except Exception as e:
        logger.error(f"Error in usage node: {str(e)}")
        return {"usage_output": f"Error: {str(e)}"}


# Routing function for conditional delegation
def route_after_orchestrator(state: WorkflowState) -> List[str]:
    """Determine which agents to route to based on orchestrator output"""
    output = state.get("orchestrator_output", "").lower()
    branches = []
    if "source" in output:
        branches.append("source")
    if "permissions" in output:
        branches.append("permissions")
    if "usage" in output:
        branches.append("usage")
    if not branches:
        return [END]
    return branches


# Create the workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("source", source_node)
workflow.add_node("permissions", permissions_node)
workflow.add_node("usage", usage_node)
workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"source": "source", "permissions": "permissions", "usage": "usage", END: END},
)
workflow.add_edge("source", END)
workflow.add_edge("permissions", END)
workflow.add_edge("usage", END)
# Compile the workflow
app_workflow = workflow.compile()

router = APIRouter()


@router.post("/process", response_model=ResponseModel)
async def process_request(request: RequestModel):
    """
    Main POST endpoint to receive and process requests
    This endpoint uses the orchestrator agent to process incoming requests.
    """
    try:
        logger.info("Processing request", request=request)
        # Use the LangGraph workflow to process the request (async)
        workflow_result = await app_workflow.ainvoke(
            {"input": request.message, "form_fields": request.formFields}
        )
        logger.info("Workflow result", result=workflow_result)
        # Synthesize outputs (simple concatenation; improve with LLM if needed)
        """ synthesized_output = "\n".join(
            [
                workflow_result.get("orchestrator_output", ""),
                workflow_result.get("source_output", ""),
                workflow_result.get("permissions_output", ""),
                workflow_result.get("usage_output", ""),
            ]
        ).strip() """
        # Process the request with workflow results (handle missing outputs if not delegated)
        processed_data: Any = {
            "received_message": request.message,
            # "orchestrator_output": workflow_result.get("orchestrator_output"),
            # "source_output": workflow_result.get("source_output"),
            # "permissions_output": workflow_result.get("permissions_output"),
            # "usage_output": workflow_result.get("usage_output"),
            # "synthesized_output": synthesized_output
            # if synthesized_output
            # else "No agent outputs generated.",
            # "received_form_fields": request.formFields,
            # "received_data": request.data,
            # "received_metadata": request.metadata,
            # "processed_at": datetime.now().isoformat(),
        }
        return ResponseModel(
            status="success",
            message="Request processed successfully by orchestrator agent",
            data=workflow_result.get("orchestrator_output"),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(
            "Error processing orchestrator request",
            error=str(e),
            error_type=type(e).__name__,
            request_message=request.message,
            form_fields_count=len(request.formFields) if request.formFields else 0,
            timestamp=datetime.now().isoformat(),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=(f"Error processing request: {str(e)}")
        )
