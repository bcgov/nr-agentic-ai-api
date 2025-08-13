from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (user, assistant, system)")
    content: str = Field(..., description="The content of the message")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for tracking")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list, 
        description="Previous messages in the conversation"
    )
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(0.7, description="Temperature for response generation")


class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI's response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID if provided")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used (if available)")