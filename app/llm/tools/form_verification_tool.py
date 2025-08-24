"""
Form verification tool for LangChain agents.
"""
from langchain.tools import tool
from app.llm.verification_agent_llm import VerificationAgentLLM
import json

@tool
def verify_fee_exemption_form(input_data: str) -> str:
    """
    Verifies a fee exemption form submission for completeness and eligibility.
    
    Args:
        input_data: JSON string that can contain either:
        - Form fields array directly: [{"data_id": "field_id", "fieldLabel": "label", "fieldType": "type", "fieldValue": "value"}]
        - Full payload with formFields: {"message": "...", "formFields": [...]}
    
    Returns:
        String containing validation results with issues and suggestions if any.
    """
    try:
        # Parse the input data
        if isinstance(input_data, str):
            parsed_data = json.loads(input_data)
        else:
            parsed_data = input_data
            
        # Extract form fields from the data
        if isinstance(parsed_data, list):
            # Direct form fields array
            form_fields = parsed_data
        elif isinstance(parsed_data, dict) and "formFields" in parsed_data:
            # Full payload with formFields
            form_fields = parsed_data["formFields"]
        else:
            return "Error: Unable to find form fields in the provided data. Expected either a form fields array or an object with 'formFields' property."
            
        if not form_fields:
            return "Error: No form fields provided for verification."
            
        # Convert form fields to the expected format if they're Pydantic models
        if hasattr(form_fields[0], 'dict'):
            form_fields = [field.dict() for field in form_fields]
            
        # Run verification
        verifier = VerificationAgentLLM()
        result = verifier.verify_form(form_fields)
        
        # Format result for LLM agent
        if result['is_valid']:
            field_details = "\n".join(result['field_analysis'])
            return f"✅ Form validation passed. The fee exemption form is complete and eligible for processing.\n\nField Analysis:\n{field_details}"
        else:
            field_details = "\n".join(result['field_analysis'])
            issues_text = "\n".join([f"• {issue}" for issue in result['issues']])
            suggestions_text = "\n".join([f"• {suggestion}" for suggestion in result['suggestions']])
            
            response = "❌ Form validation failed.\n\n"
            response += f"Field Analysis:\n{field_details}\n\n"
            
            if result['issues']:
                response += f"Critical Issues:\n{issues_text}\n\n"
            if result['suggestions']:
                response += f"Recommendations:\n{suggestions_text}"
                
            return response
            
    except Exception as e:
        return f"Error verifying form: {str(e)}"
