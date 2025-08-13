"""
Main FastAPI application with POST endpoint backbone
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Initialize FastAPI app
app = FastAPI(
    title="NR Agentic AI API",
    description=(
        "An agentic AI API built with FastAPI, LangGraph, and LangChain"
    ),
    version="0.1.0"
)


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
    
    This is the backbone endpoint that will handle incoming requests
    and can be extended with specific processing logic.
    """
    try:
        # Print the form fields data
        if request.formFields:
            print("=== FORM FIELDS DATA ===")
            for i, field in enumerate(request.formFields):
                print(f"Field {i + 1}:")
                print(f"  data-id: {field.data_id}")
                print(f"  fieldLabel: {field.fieldLabel}")
                print(f"  fieldType: {field.fieldType}")
                print(f"  fieldValue: {field.fieldValue}")
                print()
        
        # Process the request (placeholder for actual processing logic)
        processed_data = {
            "received_message": request.message,
            "received_form_fields": request.formFields,
            "received_data": request.data,
            "received_metadata": request.metadata,
            "processed_at": datetime.now().isoformat()
        }
        
        return ResponseModel(
            status="success",
            message="Request processed successfully",
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
