from math import log
import os
from datetime import datetime
from typing import List, Optional, TypedDict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


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


# Base response model
class ResponseModel(BaseModel):
    """Base response model for the POST endpoint"""

    status: str
    message: str


@router.post("/process", response_model=ResponseModel)
async def process_request(request: RequestModel):
    """
    Main POST endpoint to receive and process requests

    This endpoint uses the orchestrator agent to process incoming requests.
    """
    try:
        logger.info(
            "Processing orchestrator request",
            request_message=request.message,
            form_fields=request.formFields,
            form_fields_count=len(request.formFields) if request.formFields else 0,
        )
        # Use the LangGraph workflow to process the request
        # workflow_result = app_workflow.invoke({"input": request.message})

        return ResponseModel(
            status="success",
            message="success",
        )

    except Exception as e:
        logger.error(
            "Error processing orchestrator request",
            error=str(e),
            error_type=type(e).__name__,
            request_message=request.message,
            form_fields_count=len(request.formFields) if request.formFields else 0,
            has_data=bool(request.data),
            has_metadata=bool(request.metadata),
            timestamp=datetime.now().isoformat(),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )
