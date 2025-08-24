"""
Tools for form processing and analysis
"""
from typing import Dict, List, Any, Optional
import re
import json

# Mock LangChain tool decorator for now - can be replaced with actual import when dependencies are available
def tool(func):
    """Mock tool decorator"""
    func.invoke = lambda kwargs: func(**kwargs)
    return func

# Import models from relative path
try:
    from ..models import FormField, ValidationError, FormAnalysisResult
except ImportError:
    # Fallback for testing - define minimal classes
    class FormAnalysisResult:
        def __init__(self, required_fields=None, missing_fields=None, field_dependencies=None, completion_percentage=0):
            self.required_fields = required_fields or []
            self.missing_fields = missing_fields or []
            self.field_dependencies = field_dependencies or {}
            self.completion_percentage = completion_percentage
    
    class ValidationError:
        def __init__(self, field_id, error_message, error_type):
            self.field_id = field_id
            self.error_message = error_message
            self.error_type = error_type


@tool
def analyze_form_structure(form_fields: List[Dict[str, Any]]) -> FormAnalysisResult:
    """
    Analyze the structure of form fields to understand requirements and dependencies.
    
    Args:
        form_fields: List of form field dictionaries
        
    Returns:
        FormAnalysisResult with analysis details
    """
    required_fields = []
    missing_fields = []
    field_dependencies = {}
    total_fields = len(form_fields)
    filled_fields = 0
    
    for field in form_fields:
        field_id = field.get("data_id", "")
        field_label = field.get("field_label", "")
        field_value = field.get("field_value", "")
        
        # Check if field is required (indicated by * in label or explicit flag)
        is_required = field.get("is_required", False) or field_label.startswith("*")
        
        if is_required:
            required_fields.append(field_id)
            if not field_value or field_value.strip() == "":
                missing_fields.append(field_id)
        
        # Count filled fields
        if field_value and field_value.strip():
            filled_fields += 1
            
        # Analyze field dependencies (example: fee exemption logic)
        if "FeeExemption" in field_id:
            if field_id == "V1IsEligibleForFeeExemption":
                field_dependencies[field_id] = ["V1FeeExemptionCategory", "V1FeeExemptionClientNumber"]
            elif field_id == "V1IsExistingExemptClient":
                field_dependencies[field_id] = ["V1FeeExemptionClientNumber"]
    
    completion_percentage = (filled_fields / total_fields * 100) if total_fields > 0 else 0
    
    return FormAnalysisResult(
        required_fields=required_fields,
        missing_fields=missing_fields,
        field_dependencies=field_dependencies,
        completion_percentage=completion_percentage
    )


@tool
def extract_information_from_message(message: str, form_fields: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract relevant information from user message that can be mapped to form fields.
    
    Args:
        message: User's message
        form_fields: List of form field dictionaries
        
    Returns:
        Dictionary mapping field IDs to extracted values
    """
    extracted_data = {}
    message_lower = message.lower()
    
    # Extract patterns for different field types
    patterns = {
        "government": r"(federal|provincial|municipal|state|local)\s*government",
        "client_number": r"client\s*(?:number|id|#)?\s*:?\s*([a-zA-Z0-9\-]+)",
        "exemption": r"(exempt|exemption)",
        "organization": r"(non-?profit|charity|first\s*nations?|indigenous)",
    }
    
    for field in form_fields:
        field_id = field.get("data_id", "")
        field_type = field.get("field_type", "")
        
        # Map based on field context and message content
        if "FeeExemptionCategory" in field_id:
            gov_match = re.search(patterns["government"], message_lower)
            if gov_match:
                gov_type = gov_match.group(1).title()
                extracted_data[field_id] = f"{gov_type} Government"
            elif re.search(patterns["organization"], message_lower):
                if "non" in message_lower and "profit" in message_lower:
                    extracted_data[field_id] = "Non-Profit"
                elif "first" in message_lower and "nation" in message_lower:
                    extracted_data[field_id] = "First Nations"
        
        elif "ClientNumber" in field_id:
            client_match = re.search(patterns["client_number"], message_lower)
            if client_match:
                extracted_data[field_id] = client_match.group(1)
        
        elif "IsEligibleForFeeExemption" in field_id:
            if re.search(patterns["exemption"], message_lower):
                extracted_data[field_id] = "Yes"
        
        elif "IsExistingExemptClient" in field_id:
            if "existing" in message_lower and re.search(patterns["exemption"], message_lower):
                extracted_data[field_id] = "Yes"
    
    return extracted_data


@tool
def validate_field_data(field_data: Dict[str, Any], validation_rules: Dict[str, Any] = None) -> List[ValidationError]:
    """
    Validate form field data against business rules and constraints.
    
    Args:
        field_data: Dictionary containing field ID and value
        validation_rules: Optional validation rules
        
    Returns:
        List of validation errors
    """
    errors = []
    validation_rules = validation_rules or {}
    
    for field_id, value in field_data.items():
        # Required field validation
        if not value or str(value).strip() == "":
            if field_id in ["V1FeeExemptionClientNumber"]:  # Known required fields
                errors.append(ValidationError(
                    field_id=field_id,
                    error_message="This field is required",
                    error_type="required"
                ))
        
        # Business logic validation
        if field_id == "V1FeeExemptionCategory" and value:
            valid_categories = ["Federal Government", "Provincial Government", "Municipal Government", "First Nations", "Non-Profit"]
            if value not in valid_categories:
                errors.append(ValidationError(
                    field_id=field_id,
                    error_message=f"Invalid category. Must be one of: {', '.join(valid_categories)}",
                    error_type="invalid_option"
                ))
    
    return errors


@tool
def generate_suggestions(form_fields: List[Dict[str, Any]], user_context: Dict[str, Any] = None) -> List[str]:
    """
    Generate helpful suggestions for form completion.
    
    Args:
        form_fields: List of form field dictionaries
        user_context: Optional user context information
        
    Returns:
        List of suggestion strings
    """
    suggestions = []
    user_context = user_context or {}
    
    # Analyze current form state
    has_fee_exemption = any(field.get("data_id") == "V1IsEligibleForFeeExemption" and 
                           field.get("field_value") == "Yes" for field in form_fields)
    
    has_existing_client = any(field.get("data_id") == "V1IsExistingExemptClient" and 
                             field.get("field_value") == "Yes" for field in form_fields)
    
    missing_client_number = any(field.get("data_id") == "V1FeeExemptionClientNumber" and 
                               not field.get("field_value") for field in form_fields)
    
    # Generate contextual suggestions
    if has_fee_exemption and missing_client_number:
        suggestions.append("Since you're eligible for fee exemption, you'll need to provide your client number.")
    
    if has_existing_client and missing_client_number:
        suggestions.append("As an existing exempt client, please enter your previously assigned client number.")
    
    # if has_fee_exemption:
    #     suggestions.append("Consider providing supporting information to strengthen your fee exemption application.")
    
    # General suggestions
    if len([f for f in form_fields if f.get("field_value")]) < len(form_fields) * 0.5:
        suggestions.append("Complete at least half of the form fields to proceed with your application.")
    
    return suggestions


@tool
def generate_clarifying_questions(missing_fields: List[str], form_fields: List[Dict[str, Any]]) -> List[str]:
    """
    Generate clarifying questions for missing or incomplete information.
    
    Args:
        missing_fields: List of field IDs that are missing values
        form_fields: List of all form field dictionaries
        
    Returns:
        List of question strings
    """
    questions = []
    field_map = {field.get("data_id"): field for field in form_fields}
    
    for field_id in missing_fields:
        field = field_map.get(field_id, {})
        field_label = field.get("field_label", "")
        
        if field_id == "V1FeeExemptionClientNumber":
            questions.append("What is your client number for the fee exemption program?")
        
        elif field_id == "V1FeeExemptionSupportingInfo":
            questions.append("Would you like to provide any supporting information for your fee exemption request?")
        
        elif field_label:
            # Generic question based on field label
            clean_label = field_label.replace("*", "").strip()
            if clean_label:
                questions.append(f"Could you please provide: {clean_label}")
    
    return questions


@tool
def calculate_confidence_score(form_fields: List[Dict[str, Any]], filled_fields: Dict[str, str], 
                              validation_errors: List[ValidationError]) -> float:
    """
    Calculate confidence score for the form filling results.
    
    Args:
        form_fields: Original form fields
        filled_fields: Fields that were filled
        validation_errors: Any validation errors
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    total_fields = len(form_fields)
    filled_count = len([f for f in form_fields if f.get("field_value")])
    auto_filled_count = len(filled_fields)
    error_count = len(validation_errors)
    
    # Base score from completion percentage
    completion_score = filled_count / total_fields if total_fields > 0 else 0
    
    # Bonus for successful auto-fill
    auto_fill_bonus = (auto_filled_count / total_fields * 0.2) if total_fields > 0 else 0
    
    # Penalty for validation errors
    error_penalty = min(error_count * 0.1, 0.3)
    
    # Calculate final confidence
    confidence = min(completion_score + auto_fill_bonus - error_penalty, 1.0)
    confidence = max(confidence, 0.0)
    
    return round(confidence, 2)
