"""
Workflow graph and node definitions for LLM agentic flow.
"""
from langgraph.graph import StateGraph, END, START
from typing import TypedDict
from .agents.land_agent import land_executor
from .agents.water_agent import water_executor

class WorkflowState(TypedDict):
    input: str
    route: str
    response: str

async def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Orchestrator node that routes to land or water agent."""
    # Dummy routing logic for now
    route = "land" if "land" in state["input"].lower() else "water"
    return {"route": route}

async def land_node(state: WorkflowState) -> WorkflowState:
    result = await land_executor.ainvoke({"input": state["input"]})
    output_text = (
        result.get("output") if isinstance(result, dict) else None
    ) or str(result)
    return {"response": output_text}

async def water_node(state: WorkflowState) -> WorkflowState:
    result = await water_executor.ainvoke({"input": state["input"]})
    output_text = (
        result.get("output") if isinstance(result, dict) else None
    ) or str(result)
    return {"response": output_text}

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

app_workflow = workflow.compile()