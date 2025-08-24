"""
FastAPI endpoints for agentic form filling
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

# Create router for agentic AI endpoints
router = APIRouter(prefix="/agenticai", tags=["Agentic AI Form Filling"])

# Import models with fallback
try:
    from app.agenticai.models import FormFillingRequest, FormFillingResponse, FormFillingResult, EXAMPLE_FORM_FIELDS
except ImportError:
    from ..models import FormFillingRequest, FormFillingResponse, FormFillingResult, EXAMPLE_FORM_FIELDS

# Simple form processing without full workflow for now
@router.post("/fill-form")
async def fill_form(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligent form filling endpoint using agentic AI workflow.
    
    This endpoint processes form filling requests using a multi-agent workflow that:
    1. Analyzes form structure and requirements
    2. Extracts information from user messages
    3. Auto-fills form fields intelligently
    4. Validates form data
    5. Generates questions for missing information
    """
    try:
        # Import tools directly
        from ..tools import (
            analyze_form_structure,
            extract_information_from_message,
            validate_field_data,
            generate_suggestions,
            generate_clarifying_questions
        )
        
        message = request.get("message", "")
        form_fields = request.get("form_fields", [])
        user_context = request.get("user_context", {})
        
        # Process the form through simplified workflow
        # 1. Analyze form structure
        analysis = analyze_form_structure(form_fields)
        
        # 2. Extract information from message
        extracted_data = extract_information_from_message(message, form_fields)
        
        # 3. Apply extracted data to form fields
        for field in form_fields:
            field_id = field.get("data_id", "")
            if field_id in extracted_data:
                field["field_value"] = extracted_data[field_id]
        
        # 4. Validate form data
        field_data = {field.get("data_id"): field.get("field_value", "") for field in form_fields}
        validation_errors = validate_field_data(field_data)
        
        # 5. Generate questions and suggestions
        missing_fields = [field.get("data_id") for field in form_fields 
                         if not field.get("field_value") and field.get("is_required")]
        
        questions = generate_clarifying_questions(missing_fields, form_fields)
        suggestions = generate_suggestions(form_fields, user_context)
        
        # Determine status
        status_result = "complete" if not missing_fields and not validation_errors else "needs_info"
        
        # Calculate confidence score
        filled_count = len([f for f in form_fields if f.get("field_value")])
        total_count = len(form_fields)
        confidence_score = filled_count / total_count if total_count > 0 else 0
        
        return {
            "status": status_result,
            "result": {
                "status": status_result,
                "filled_fields": extracted_data,
                "missing_information": missing_fields,
                "validation_errors": [{"field_id": e.field_id, "error_message": e.error_message, "error_type": e.error_type} for e in validation_errors],
                "suggestions": suggestions,
                "questions_for_user": questions,
                "confidence_score": confidence_score
            },
            "workflow_state": "complete",
            "next_action": "Please provide the requested information to complete the form" if status_result == "needs_info" else "Form is ready for submission"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing form: {str(e)}"
        )


@router.get("/example-form")
async def get_example_form() -> Dict[str, Any]:
    """
    Get an example form structure for testing the form filling endpoint.
    
    Returns:
        Dictionary containing example form fields and usage instructions
    """
    return {
        "message": "Can you please help me fill this water licence application form, please verify if all fields are correct or you need more information",
        "form_fields": EXAMPLE_FORM_FIELDS,
        "user_context": {
            "organization_type": "government",
            "previous_applications": False,
            "urgency_level": "normal"
        },
        "usage_instructions": {
            "endpoint": "/agenticai/fill-form",
            "method": "POST",
            "description": "Send this example payload to test the agentic form filling system",
            "expected_behavior": [
                "The system will analyze the form structure",
                "Extract relevant information from the message", 
                "Auto-fill appropriate fields",
                "Validate the form data",
                "Generate questions for missing information",
                "Provide suggestions for form completion"
            ]
        }
    }


@router.get("/workflow-info")
async def get_workflow_info() -> Dict[str, Any]:
    """
    Get information about the agentic form filling workflow.
    
    Returns:
        Dictionary describing the workflow stages and capabilities
    """
    return {
        "workflow_name": "Intelligent Form Filling Workflow",
        "description": "Multi-agent system for intelligent form processing and auto-completion",
        "agents": {
            "form_analyzer": {
                "purpose": "Analyze form structure, identify required fields, and understand dependencies",
                "capabilities": ["Field requirement analysis", "Dependency mapping", "Completion tracking"]
            },
            "information_extractor": {
                "purpose": "Extract relevant information from user messages",
                "capabilities": ["Natural language processing", "Entity extraction", "Context understanding"]
            },
            "auto_fill_agent": {
                "purpose": "Intelligently populate form fields",
                "capabilities": ["Smart field mapping", "Business logic application", "Value suggestion"]
            },
            "validation_agent": {
                "purpose": "Validate form data and check business rules",
                "capabilities": ["Field validation", "Business rule checking", "Error detection"]
            },
            "question_generator": {
                "purpose": "Generate clarifying questions for missing information",
                "capabilities": ["Smart question generation", "Context-aware prompts", "User guidance"]
            }
        },
        "workflow_steps": [
            "1. Form Analysis - Understand structure and requirements",
            "2. Information Extraction - Parse user input for relevant data",
            "3. Auto-Fill - Populate fields based on extracted information",
            "4. Validation - Check data completeness and correctness", 
            "5. Question Generation - Create questions for missing information"
        ],
        "supported_field_types": [
            "text", "textarea", "radio", "select-one", "checkbox", "email", "number", "date"
        ],
        "features": [
            "Intelligent auto-completion",
            "Context-aware field mapping",
            "Business rule validation",
            "Missing information detection",
            "Smart question generation",
            "Confidence scoring",
            "Multi-step workflow processing"
        ]
    }


@router.post("/validate-form")
async def validate_form_only(request: FormFillingRequest) -> Dict[str, Any]:
    """
    Validate form data without auto-filling (validation-only mode).
    
    Args:
        request: FormFillingRequest containing form fields to validate
        
    Returns:
        Dictionary with validation results only
    """
    try:
        # Import validation tools
        from app.agenticai.tools import validate_field_data, analyze_form_structure
        
        # Convert form fields to dict format
        form_fields_dict = [field.dict() for field in request.form_fields]
        
        # Analyze form structure
        analysis = analyze_form_structure.invoke({"form_fields": form_fields_dict})
        
        # Validate current field data
        field_data = {field.data_id: field.field_value for field in request.form_fields}
        validation_errors = validate_field_data.invoke({"field_data": field_data})
        
        return {
            "status": "validation_complete",
            "analysis": analysis.dict(),
            "validation_errors": [error.dict() for error in validation_errors],
            "completion_percentage": analysis.completion_percentage,
            "required_fields_missing": len(analysis.missing_fields),
            "total_errors": len(validation_errors)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating form: {str(e)}"
        )
