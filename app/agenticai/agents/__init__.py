"""
Individual agents for form processing workflow
"""
from typing import Dict, Any, List
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.agenticai.models import AgentState, FormAnalysisResult, ValidationError
from app.agenticai.tools import (
    analyze_form_structure, 
    extract_information_from_message,
    validate_field_data,
    generate_suggestions,
    generate_clarifying_questions,
    calculate_confidence_score
)


class FormAnalyzerAgent:
    """Agent responsible for analyzing form structure and requirements"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a form analysis expert. Your job is to analyze form fields and understand:
            1. Which fields are required vs optional
            2. Field dependencies and relationships
            3. Overall form completion status
            4. Business logic and validation requirements
            
            Analyze the form carefully and provide structured insights."""),
            ("human", "Analyze this form: {form_data}")
        ])
    
    def analyze(self, state: AgentState) -> AgentState:
        """Analyze form structure and update state"""
        try:
            # Convert form fields to dict format for tools
            form_fields_dict = [field.dict() for field in state.form_fields]
            
            # Use the analysis tool
            analysis_result = analyze_form_structure.invoke({"form_fields": form_fields_dict})
            
            # Update state with analysis results
            state.analysis_result = analysis_result
            state.missing_information = analysis_result.missing_fields
            state.current_step = "extract_info"
            
            return state
            
        except Exception as e:
            print(f"Error in form analysis: {e}")
            state.current_step = "error"
            return state


class InformationExtractionAgent:
    """Agent responsible for extracting information from user messages"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured information from natural language.
            Your job is to:
            1. Parse user messages for relevant form data
            2. Map extracted information to appropriate form fields
            3. Handle context and implied information
            4. Extract entities like names, numbers, categories, etc.
            
            Be precise and only extract information you're confident about."""),
            ("human", "Extract form data from this message: {message}\nFor these form fields: {form_fields}")
        ])
    
    def extract(self, state: AgentState) -> AgentState:
        """Extract information from user message and update state"""
        try:
            # Convert form fields to dict format
            form_fields_dict = [field.dict() for field in state.form_fields]
            
            # Extract information using the tool
            extracted_data = extract_information_from_message.invoke({
                "message": state.user_message,
                "form_fields": form_fields_dict
            })
            
            # Update filled fields
            state.filled_fields.update(extracted_data)
            state.current_step = "auto_fill"
            
            return state
            
        except Exception as e:
            print(f"Error in information extraction: {e}")
            state.current_step = "error"
            return state


class AutoFillAgent:
    """Agent responsible for intelligently filling form fields"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent form auto-fill agent. Your job is to:
            1. Use extracted information to populate form fields
            2. Apply business logic and field dependencies
            3. Make intelligent suggestions for missing data
            4. Ensure data consistency across related fields
            
            Only fill fields when you have high confidence in the values."""),
            ("human", "Auto-fill form with this data: {extracted_data}\nForm fields: {form_fields}")
        ])
    
    def auto_fill(self, state: AgentState) -> AgentState:
        """Auto-fill form fields based on extracted information"""
        try:
            # Apply auto-fill logic to form fields
            for field in state.form_fields:
                field_id = field.data_id
                
                # If we have extracted data for this field, update it
                if field_id in state.filled_fields:
                    field.field_value = state.filled_fields[field_id]
                
                # Apply business logic for dependent fields
                self._apply_business_logic(field, state)
            
            state.current_step = "validate"
            return state
            
        except Exception as e:
            print(f"Error in auto-fill: {e}")
            state.current_step = "error"
            return state
    
    def _apply_business_logic(self, field, state: AgentState):
        """Apply business logic for field dependencies"""
        # Example: If eligible for fee exemption but not existing client, 
        # suggest they need to provide supporting information
        if field.data_id == "V1FeeExemptionSupportingInfo":
            eligible = any(f.data_id == "V1IsEligibleForFeeExemption" and f.field_value == "Yes" 
                          for f in state.form_fields)
            existing_client = any(f.data_id == "V1IsExistingExemptClient" and f.field_value == "Yes" 
                                 for f in state.form_fields)
            
            if eligible and not existing_client and not field.field_value:
                state.suggestions.append(
                    "Since you're eligible for fee exemption but not an existing client, "
                    "providing supporting information would strengthen your application."
                )


class ValidationAgent:
    """Agent responsible for validating form data"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def validate(self, state: AgentState) -> AgentState:
        """Validate form data and update state"""
        try:
            # Prepare field data for validation
            field_data = {field.data_id: field.field_value for field in state.form_fields}
            
            # Validate using the tool
            validation_errors = validate_field_data.invoke({"field_data": field_data})
            
            # Update state with validation results
            state.validation_errors = validation_errors
            
            # Check if we need more information
            missing_fields = [field.data_id for field in state.form_fields 
                            if not field.field_value and field.is_required]
            
            if missing_fields or validation_errors:
                state.current_step = "ask_questions"
            else:
                state.current_step = "complete"
                state.completion_status = "complete"
            
            return state
            
        except Exception as e:
            print(f"Error in validation: {e}")
            state.current_step = "error"
            return state


class QuestionGenerationAgent:
    """Agent responsible for generating clarifying questions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that generates clear, specific questions 
            to gather missing information for form completion. Your questions should be:
            1. Clear and easy to understand
            2. Specific to the missing information
            3. Contextually relevant
            4. Polite and professional
            
            Generate questions that will help complete the form efficiently."""),
            ("human", "Generate questions for missing fields: {missing_fields}\nForm context: {form_fields}")
        ])
    
    def generate_questions(self, state: AgentState) -> AgentState:
        """Generate questions for missing information"""
        try:
            # Find missing required fields
            missing_fields = [field.data_id for field in state.form_fields 
                            if not field.field_value and field.is_required]
            
            # Generate questions using the tool
            form_fields_dict = [field.dict() for field in state.form_fields]
            questions = generate_clarifying_questions.invoke({
                "missing_fields": missing_fields,
                "form_fields": form_fields_dict
            })
            
            # Generate suggestions
            suggestions = generate_suggestions.invoke({
                "form_fields": form_fields_dict,
                "user_context": state.user_context
            })
            
            # Calculate confidence score
            confidence = calculate_confidence_score.invoke({
                "form_fields": form_fields_dict,
                "filled_fields": state.filled_fields,
                "validation_errors": state.validation_errors
            })
            
            # Update state
            state.questions_for_user = questions
            state.suggestions = suggestions
            state.confidence_score = confidence
            state.current_step = "complete"
            state.completion_status = "needs_info"
            
            return state
            
        except Exception as e:
            print(f"Error in question generation: {e}")
            state.current_step = "error"
            return state
