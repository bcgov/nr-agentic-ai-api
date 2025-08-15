from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
from app.core.logging import get_logger
import json

# Imports for agent invoke functions
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
    orchestrator_output: Dict[str, Any]
    source_output: Dict[str, Any]
    permissions_output: Dict[str, Any]
    usage_output: Dict[str, Any]
    intermediate_steps: List[Any]
    processing_summary: Dict[str, Any]  # Track which agents used LLM enhancement


# Define workflow nodes (async for compatibility)
async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that decides delegation with robust JSON parsing"""
    try:
        result = await orchestrator_executor.ainvoke({"input": state["input"]})

        # Extract the output and intermediate steps
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        # Parse the JSON output from the orchestrator
        try:
            # The output should be valid JSON
            if isinstance(output, str):
                # Try to extract JSON from the output if it's wrapped in text
                import re

                json_match = re.search(r"\{.*\}", output, re.DOTALL)
                if json_match:
                    output_json = json.loads(json_match.group())
                else:
                    # Fallback parsing
                    output_json = json.loads(output)
            else:
                output_json = output

            # Ensure required fields exist with defaults
            orchestrator_result = {
                "route": output_json.get("route", []),
                "clarifications": output_json.get("clarifications", []),
                "analysis": output_json.get("analysis", "No analysis provided"),
                "intermediate_steps": intermediate_steps,
                "raw_output": output,
            }

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse orchestrator JSON output: {e}")
            # Fallback: create a default routing based on keywords in output
            output_lower = str(output).lower()
            fallback_routes = []
            if any(
                word in output_lower for word in ["source", "water source", "intake"]
            ):
                fallback_routes.append("source")
            if any(word in output_lower for word in ["usage", "purpose", "use"]):
                fallback_routes.append("usage")
            if any(
                word in output_lower
                for word in ["permission", "permit", "license", "compliance"]
            ):
                fallback_routes.append("permissions")

            orchestrator_result = {
                "route": fallback_routes,
                "clarifications": ["Please provide more specific information"],
                "analysis": f"Fallback parsing applied due to JSON error: {e}",
                "intermediate_steps": intermediate_steps,
                "raw_output": output,
                "parsing_error": str(e),
            }

        return {"orchestrator_output": orchestrator_result}

    except Exception as e:
        logger.error(f"Error in orchestrator node: {str(e)}", exc_info=True)
        return {
            "orchestrator_output": {
                "route": [],
                "clarifications": [f"System error: {str(e)}"],
                "analysis": f"Error in orchestrator: {str(e)}",
                "intermediate_steps": [],
                "error": str(e),
            }
        }


async def source_node(state: WorkflowState) -> WorkflowState:
    """Source node that processes the request"""
    try:
        result = await invoke_source_agent(state["input"], state["form_fields"])
        return {"source_output": result}
    except Exception as e:
        logger.error(f"Error in source node: {str(e)}", exc_info=True)
        return {
            "source_output": {
                "agent": "SourceAgent",
                "status": "error",
                "error": str(e),
                "message": f"Error: {str(e)}",
            }
        }


async def permissions_node(state: WorkflowState) -> WorkflowState:
    """Permissions node that processes the request"""
    try:
        result = await invoke_permissions_agent(state["input"], state["form_fields"])
        return {"permissions_output": result}
    except Exception as e:
        logger.error(f"Error in permissions node: {str(e)}", exc_info=True)
        return {
            "permissions_output": {
                "agent": "PermissionsAgent",
                "status": "error",
                "error": str(e),
                "message": f"Error: {str(e)}",
            }
        }


async def usage_node(state: WorkflowState) -> WorkflowState:
    """Usage node that processes the request"""
    try:
        result = await invoke_usage_agent(state["input"], state["form_fields"])
        return {"usage_output": result}
    except Exception as e:
        logger.error(f"Error in usage node: {str(e)}", exc_info=True)
        return {
            "usage_output": {
                "agent": "UsageAgent",
                "status": "error",
                "error": str(e),
                "message": f"Error: {str(e)}",
            }
        }


# Routing function for conditional delegation
def route_after_orchestrator(state: WorkflowState) -> List[str]:
    """Determine which agents to route to based on orchestrator JSON output"""
    try:
        orchestrator_output = state.get("orchestrator_output", {})

        if isinstance(orchestrator_output, dict):
            routes = orchestrator_output.get("route", [])

            # Map route names to node names
            route_mapping = {
                "source": "source",
                "usage": "usage",
                "permissions": "permissions",
            }

            branches = []
            for route in routes:
                if route in route_mapping:
                    branches.append(route_mapping[route])

            # If no explicit routes, default to all agents for comprehensive analysis
            if not branches:
                logger.info("No specific routes found, defaulting to all agents")
                branches = ["source", "usage", "permissions"]

            logger.info(f"Routing to agents: {branches}")
            return branches if branches else [END]

    except Exception as e:
        logger.error(f"Error in routing: {e}")
        # Fallback to all agents on routing error
        return ["source", "usage", "permissions"]

    return [END]


# Create the workflow with parallel execution capability
workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("source", source_node)
workflow.add_node("permissions", permissions_node)
workflow.add_node("usage", usage_node)
workflow.set_entry_point("orchestrator")

# Enable parallel routing to multiple agents
workflow.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {"source": "source", "permissions": "permissions", "usage": "usage", END: END},
)

# All agents flow to END (they can execute in parallel)
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
    This endpoint uses the orchestrator agent to process incoming requests with parallel execution.
    """
    try:
        logger.info(
            "Processing orchestrator request",
            request_message=request.message,
            form_fields_count=len(request.formFields) if request.formFields else 0,
        )

        # Use the LangGraph workflow to process the request (async with parallel execution)
        workflow_result = await app_workflow.ainvoke(
            {"input": request.message, "form_fields": request.formFields}
        )

        logger.info("Workflow completed", result_keys=list(workflow_result.keys()))

        # Ensure all outputs are present, using "skipped" for missing ones
        def get_output_or_skipped(key: str) -> Dict[str, Any]:
            output = workflow_result.get(key)
            if output is None:
                return {
                    "agent": key.replace("_output", "").title() + "Agent",
                    "status": "skipped",
                    "message": "Agent was not routed to based on orchestrator decision",
                    "query": request.message,
                }
            return output

        # Structure the comprehensive response with processing method tracking
        def extract_processing_info(output: Dict[str, Any]) -> Dict[str, str]:
            if isinstance(output, dict) and output.get("status") != "skipped":
                return {
                    "method": output.get("processing_method", "unknown"),
                    "enhanced": "yes" if output.get("enhanced_analysis") else "no",
                }
            return {"method": "skipped", "enhanced": "no"}

        processing_summary = {
            "source": extract_processing_info(get_output_or_skipped("source_output")),
            "usage": extract_processing_info(get_output_or_skipped("usage_output")),
            "permissions": extract_processing_info(
                get_output_or_skipped("permissions_output")
            ),
            "total_llm_enhanced": sum(
                1
                for info in [
                    extract_processing_info(get_output_or_skipped("source_output")),
                    extract_processing_info(get_output_or_skipped("usage_output")),
                    extract_processing_info(
                        get_output_or_skipped("permissions_output")
                    ),
                ]
                if info["enhanced"] == "yes"
            ),
        }

        processed_data = {
            "orchestrator_output": workflow_result.get(
                "orchestrator_output",
                {
                    "route": [],
                    "clarifications": ["No orchestrator output available"],
                    "analysis": "Orchestrator did not execute properly",
                },
            ),
            "source_output": get_output_or_skipped("source_output"),
            "permissions_output": get_output_or_skipped("permissions_output"),
            "usage_output": get_output_or_skipped("usage_output"),
            "workflow_summary": {
                "total_agents": 3,
                "executed_agents": sum(
                    1
                    for key in ["source_output", "permissions_output", "usage_output"]
                    if workflow_result.get(key) is not None
                ),
                "skipped_agents": sum(
                    1
                    for key in ["source_output", "permissions_output", "usage_output"]
                    if workflow_result.get(key) is None
                ),
                "llm_enhanced_agents": processing_summary["total_llm_enhanced"],
                "rule_based_agents": sum(
                    1
                    for info in processing_summary.values()
                    if isinstance(info, dict) and info.get("method") == "rule_based"
                ),
                "has_errors": any(
                    output.get("status") == "error"
                    for output in [
                        workflow_result.get("source_output", {}),
                        workflow_result.get("permissions_output", {}),
                        workflow_result.get("usage_output", {}),
                    ]
                    if isinstance(output, dict)
                ),
                "processing_methods": processing_summary,
            },
            "intermediate_steps": workflow_result.get("orchestrator_output", {}).get(
                "intermediate_steps", []
            ),
            "processed_at": datetime.now().isoformat(),
        }

        return ResponseModel(
            status="success",
            message="Request processed successfully by agentic workflow",
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
            timestamp=datetime.now().isoformat(),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
