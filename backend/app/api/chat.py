"""Chat endpoint wiring conversational requests to Azure OpenAI."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import (
    get_chat_client,
    get_cosmos_logger,
    get_retrieval_augmentor,
)
from ..models.forms import ChatMessage, ChatRequest, ChatResponse
from ..services.azure_clients import (
    AzureCosmosLogger,
    AzureOpenAIChatClient,
    RetrievalAugmentor,
)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_client: AzureOpenAIChatClient = Depends(get_chat_client),
    retrieval: RetrievalAugmentor = Depends(get_retrieval_augmentor),
    cosmos_logger: AzureCosmosLogger = Depends(get_cosmos_logger),
):
    """Generate a grounded response for the provided prompt."""

    system_message = ChatMessage(
        role="system",
        content=(
            "You are the agentic assistant helping applicants complete the "
            "British Columbia water licence forms. Provide concise guidance, "
            "grounded in regulations and schema metadata."
        ),
    )
    conversation = [system_message, *payload.history]
    if payload.prompt:
        conversation.append(ChatMessage(role="user", content=payload.prompt))

    grounding_chunks = await retrieval.search(
        payload.prompt or "", page_id=payload.page.id
    )
    completion, sources = await chat_client.generate_reply(conversation, grounding_chunks)

    cosmos_logger.log_chat(
        session=payload.session.dict(),
        page=payload.page.dict(),
        prompt=payload.prompt,
        response=completion.content,
        sources=sources,
    )

    return ChatResponse(messages=[ChatMessage(role=completion.role, content=completion.content)], sources=sources)
