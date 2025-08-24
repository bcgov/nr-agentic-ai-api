"""
Pydantic models for agentic AI form processing
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class FieldType(str, Enum):
    """Supported form field types"""
    TEXT = "text"
    TEXTAREA = "textarea"
    RADIO = "radio"
    SELECT_ONE = "select-one"
    CHECKBOX = "checkbox"
    EMAIL = "email"
    NUMBER = "number"
    DATE = "date"


class FormField(BaseModel):
    """Individual form field model"""
    data_id: str = Field(..., description="Unique identifier for the field")
    field_label: str = Field(default="", description="Display label for the field")
    field_type: FieldType = Field(..., description="Type of the form field")
    field_value: Optional[str] = Field(default="", description="Current value of the field")
    is_required: bool = Field(default=False, description="Whether the field is required")
    validation_rules: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation rules for the field")
    options: Optional[List[str]] = Field(default_factory=list, description="Available options for select/radio fields")


class FormFillingRequest(BaseModel):
    """Request model for form filling endpoint"""
    message: str = Field(..., description="User message requesting form assistance")
    form_fields: List[FormField] = Field(..., description="List of form fields to process")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional user context")


class FormAnalysisResult(BaseModel):
    """Result of form analysis"""
    required_fields: List[str] = Field(default_factory=list, description="List of required field IDs")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required field IDs")
    field_dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="Field dependencies mapping")
    completion_percentage: float = Field(default=0.0, description="Form completion percentage")


class ValidationError(BaseModel):
    """Validation error model"""
    field_id: str = Field(..., description="Field ID with validation error")
    error_message: str = Field(..., description="Validation error message")
    error_type: str = Field(..., description="Type of validation error")


class FormFillingResult(BaseModel):
    """Result of form filling process"""
    status: str = Field(..., description="Processing status")
    filled_fields: Dict[str, str] = Field(default_factory=dict, description="Auto-filled field values")
    missing_information: List[str] = Field(default_factory=list, description="Information still needed from user")
    validation_errors: List[ValidationError] = Field(default_factory=list, description="Validation errors found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for form completion")
    questions_for_user: List[str] = Field(default_factory=list, description="Questions to ask the user")
    confidence_score: float = Field(default=0.0, description="Confidence in the auto-fill results")


class AgentState(BaseModel):
    """State model for the agentic workflow"""
    form_fields: List[FormField] = Field(default_factory=list, description="Form fields being processed")
    user_message: str = Field(default="", description="Original user message")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context information")
    analysis_result: Optional[FormAnalysisResult] = None
    filled_fields: Dict[str, str] = Field(default_factory=dict, description="Fields that have been filled")
    missing_information: List[str] = Field(default_factory=list, description="Missing information needed")
    validation_errors: List[ValidationError] = Field(default_factory=list, description="Current validation errors")
    questions_for_user: List[str] = Field(default_factory=list, description="Questions to ask user")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for user")
    conversation_history: List[str] = Field(default_factory=list, description="Conversation history")
    current_step: str = Field(default="analyze", description="Current step in the workflow")
    completion_status: str = Field(default="in_progress", description="Overall completion status")
    confidence_score: float = Field(default=0.0, description="Overall confidence score")


class FormFillingResponse(BaseModel):
    """Response model for form filling endpoint"""
    status: str = Field(..., description="Processing status")
    result: FormFillingResult = Field(..., description="Form filling results")
    workflow_state: str = Field(..., description="Current workflow state")
    next_action: Optional[str] = Field(None, description="Suggested next action")


# Example form fields for documentation
EXAMPLE_FORM_FIELDS = [
    {
        "data_id": "V1IsEligibleForFeeExemption",
        "field_label": "Are you eligible for fee exemption?",
        "field_type": "radio",
        "field_value": "Yes",
        "is_required": True,
        "options": ["Yes", "No"]
    },
    {
        "data_id": "V1IsExistingExemptClient",
        "field_label": "Are you an existing exempt client?",
        "field_type": "radio",
        "field_value": "Yes",
        "is_required": True,
        "options": ["Yes", "No"]
    },
    {
        "data_id": "V1FeeExemptionClientNumber",
        "field_label": "*Please enter your client number:",
        "field_type": "text",
        "field_value": "",
        "is_required": True,
        "validation_rules": {"min_length": 5, "max_length": 20}
    },
    {
        "data_id": "V1FeeExemptionCategory",
        "field_label": "*Fee Exemption Category:",
        "field_type": "select-one",
        "field_value": "Federal Government",
        "is_required": True,
        "options": ["Federal Government", "Provincial Government", "Municipal Government", "First Nations", "Non-Profit"]
    },
    {
        "data_id": "V1FeeExemptionSupportingInfo",
        "field_label": "Please enter any supporting information that will assist in determining your eligibility for a fee exemption.",
        "field_type": "textarea",
        "field_value": "",
        "is_required": False,
        "validation_rules": {"max_length": 1000}
    }
]
