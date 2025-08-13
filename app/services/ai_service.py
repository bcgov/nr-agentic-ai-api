import httpx
from typing import Dict, Any, Optional, List
from langchain_openai import AzureChatOpenAI
from app.core.config import settings


class AIService:
    def __init__(self):
        # Configure httpx client for tunnel if needed
        self.http_client = None
        if settings.USE_LOCAL_TUNNEL and settings.TUNNEL_RESOLVE_HOST:
            # Create custom httpx client with DNS resolution for tunnel
            self.http_client = httpx.AsyncClient(
                verify=False,  # Skip SSL verification for local tunnel
                headers={
                    "Host": settings.TUNNEL_RESOLVE_HOST
                }
            )
        
        # Initialize Azure OpenAI client
        self.llm = AzureChatOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.azure_openai_base_url,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=1000,
            http_async_client=self.http_client
        )
    
    async def generate_response(
        self, 
        message: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response using Azure OpenAI with conversation history"""
        try:
            messages = []
            
            # Add system context if provided
            if context and context.get("system_prompt"):
                messages.append(("system", context["system_prompt"]))
            
            # Add conversation history if provided
            if conversation_history:
                for hist_msg in conversation_history:
                    role = hist_msg.get("role", "user")
                    content = hist_msg.get("content", "")
                    
                    if role == "system":
                        messages.append(("system", content))
                    elif role == "assistant":
                        messages.append(("assistant", content))
                    else:
                        messages.append(("user", content))
            
            # Add current user message
            messages.append(("user", message))
            
            # Update LLM parameters if provided
            llm_kwargs = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if max_tokens is not None:
                llm_kwargs["max_tokens"] = max_tokens
            
            # Create a new LLM instance with updated parameters if needed
            if llm_kwargs:
                llm = AzureChatOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    azure_endpoint=settings.azure_openai_base_url,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    temperature=llm_kwargs.get("temperature", 0.7),
                    max_tokens=llm_kwargs.get("max_tokens", 1000),
                    http_async_client=self.http_client
                )
            else:
                llm = self.llm
            
            # Generate response
            response = await llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            raise Exception(f"Error generating AI response: {str(e)}")
    
    async def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """Handle chat completion with message history"""
        try:
            # Convert messages to LangChain format
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    formatted_messages.append(("system", content))
                elif role == "assistant":
                    formatted_messages.append(("assistant", content))
                else:
                    formatted_messages.append(("user", content))
            
            response = await self.llm.ainvoke(formatted_messages)
            return response.content
            
        except Exception as e:
            raise Exception(f"Error in chat completion: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        if self.http_client:
            await self.http_client.aclose()


# Global instance
ai_service = AIService()