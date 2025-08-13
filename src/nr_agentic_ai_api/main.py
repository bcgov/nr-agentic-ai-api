"""
Main FastAPI application with POST endpoint backbone
"""

import os
from typing import Any, Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate

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


# Create a simple tool for demonstration
class SimpleTool(BaseTool):
    name = "simple_tool"
    description = "A simple tool that returns the input message"
    
    def _run(self, message: str) -> str:
        return f"Processed: {message}"
    
    async def _arun(self, message: str) -> str:
        return self._run(message)


# Create the orchestrator agent
tools = [SimpleTool()]
prompt = PromptTemplate.from_template(
    "You are an orchestrator agent. Use the available tools to process the "
    "user's request.\n\nUser request: {input}\n\n{agent_scratchpad}"
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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
    data: Optional[Dict[str, Any]] = None
    timestamp: str


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
        # Use the orchestrator agent to process the request
        agent_response = await agent_executor.ainvoke(
            {"input": request.message}
        )
        
        # Process the request with agent response
        processed_data = {
            "received_message": request.message,
            "agent_response": agent_response["output"],
            "received_form_fields": request.formFields,
            "received_data": request.data,
            "received_metadata": request.metadata,
            "processed_at": datetime.now().isoformat()
        }
        
        return ResponseModel(
            status="success",
            message="Request processed successfully by orchestrator agent",
            data=processed_data,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Error processing request: {str(e)}"
            )
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
