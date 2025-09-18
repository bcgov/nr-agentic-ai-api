from uuid import uuid4

from fastapi import FastAPI, Header
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

@app.get("/health/retrieval")
def health_retrieval():
    try:
        r = retrieve("ping")
        fallback = bool(r.get("fallback")) if isinstance(r, dict) else False
        if fallback:
            error_message = r.get("error") if isinstance(r, dict) else None
            return JSONResponse(
                {"ok": False, "error": error_message or "retrieval in fallback mode"},
                status_code=503,
            )
        ok = bool(r and isinstance(r.get("snippets"), list))
        return JSONResponse({"ok": ok})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)
