from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict
from app.models.chat import ChatRequest, ChatResponse
from app.services.ai_service import ai_service

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the AI agent"""
    try:
        response = await ai_service.generate_response(
            message=request.message,
            conversation_history=request.conversation_history,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return ChatResponse(
            response=response,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@router.get("/history/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history for a specific conversation"""
    # This would typically fetch from a database
    # For now, return a placeholder response
    return {
        "conversation_id": conversation_id,
        "history": [],
        "message": "Conversation history retrieval not yet implemented"
    }