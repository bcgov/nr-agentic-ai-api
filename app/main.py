"""
Main FastAPI application with POST endpoint backbone
"""

import os
import logging
from typing import Optional, List, TypedDict
from fastapi import FastAPI, HTTPException, Header
from app.formfiller.api import router as api_router
from app.api import router as api_router_orchestrator
from pydantic import BaseModel
from dotenv import load_dotenv
from app.llm.workflow import app_workflow
from uuid import uuid4

#from app.backend.schemas import OrchestratorRequest, OrchestratorResponse, Reference
#from app.backend.graph import build_graph
from typing import Any, Dict, List


# Minimal request/response models for FastAPI endpoint typing
class RequestModel(BaseModel):
    message: str
    formFields: Optional[List[dict]] = None

class ResponseModel(BaseModel):
    status: str
    message: str
    formFields: Optional[List[dict]] = None

load_dotenv()

# Configure logging (honor LOG_LEVEL env var; default INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, LOG_LEVEL, logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(_level)

# Ensure at least one handler is configured so logs are emitted when launched via uv/uvicorn
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# Initialize FastAPI app
app = FastAPI(
    title="NR Agentic AI API",
    description=(
        "An agentic AI API built with FastAPI, LangGraph, and LangChain. "
        "Features intelligent form filling and multi-agent workflows."
    ),
    version="0.1.0"
)

# Log app init once
logger.info("NR Agentic AI API initialized (log level=%s)", LOG_LEVEL)

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
        # Log the incoming request (INFO so it appears by default)
        logger.info(
            "Processing message len=%d; formFields=%s",
            len(request.message) if request.message else 0,
            "yes" if request.formFields else "no",
        )
        if request.formFields:
            logger.info("Form fields count: %d", len(request.formFields))
        # Use the LangGraph workflow (async) to process the request
        workflow_result = await app_workflow.ainvoke({"message": request.message, "formFields": request.formFields or []})

        response = workflow_result["response"]
        # response is a dict with keys 'message' and 'formFields'
        return ResponseModel(
            status="success",
            message=response.get("message", ""),
            formFields=response.get("formFields", None)
        )
        
    except Exception as e:
        logger.exception("Unhandled error in /api/process: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        ) from e


from fastapi.responses import JSONResponse
from app.src.models import (
    ConversationRequest,
    ConversationResponse,
    OrchestratorRequest,
    OrchestratorResponse,
)
from app.src.memory.session_store import (
    ConversationState,
    get_conversation_state,
    set_conversation_state,
)
from app.src.orchestrator.conversation_adapter import (
    build_conversation_response,
    prepare_conversation,
)
from app.src.orchestrator.orchestrator import orchestrate
from app.src.retrieval.client import retrieve

app = FastAPI(title="WLRS Orchestrator")

@app.post("/orchestrate", response_model=OrchestratorResponse)
def post_orchestrate(
    req: OrchestratorRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    # Thread session id to orchestrator so agents can use Redis-backed context
    return orchestrate(req, session_id=x_session_id)


@app.post("/orchestrate-conversation", response_model=ConversationResponse)
def post_orchestrate_conversation(
    req: ConversationRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    session_identifier = x_session_id or req.thread_id
    if session_identifier:
        state = get_conversation_state(session_identifier)
    else:
        state = ConversationState()

    thread_id = session_identifier or state.thread_id or str(uuid4())
    state.thread_id = thread_id

    orch_req, history, appended_user, descriptors = prepare_conversation(req, state)
    orch_resp = orchestrate(orch_req, session_id=thread_id)
    conv_resp, new_state = build_conversation_response(
        orch_resp, history, appended_user, descriptors, thread_id=thread_id
    )
    new_state.thread_id = thread_id
    set_conversation_state(thread_id, new_state)
    return conv_resp

# Include API router
app.include_router(api_router, prefix="/api")
app.include_router(api_router_orchestrator, prefix="/api/orchestrator")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())