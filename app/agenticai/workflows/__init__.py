"""
Workflow orchestrator for agentic form filling using LangGraph
"""
from typing import Dict, Any, Literal
import asyncio

# Mock LangGraph imports for now - can be replaced when dependencies are available
class StateGraph:
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = []
        self.entry_point = None
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def add_edge(self, from_node, to_node):
        self.edges.append((from_node, to_node))
    
    def add_conditional_edges(self, from_node, condition_func, mapping):
        pass
    
    def set_entry_point(self, node):
        self.entry_point = node
    
    def compile(self):
        return MockCompiledGraph(self)

class MockCompiledGraph:
    def __init__(self, graph):
        self.graph = graph
    
    async def ainvoke(self, state):
        return await self._process_workflow(state)
    
    async def _process_workflow(self, state):
        # Simple mock workflow processing
        try:
            # Simulate workflow steps
            state.current_step = "analyze"
            state.completion_status = "needs_info"
            state.confidence_score = 0.75
            state.questions_for_user = ["What is your client number?"]
            state.suggestions = ["Provide your client number to complete the form"]
            return state
        except Exception as e:
            state.current_step = "error"
            state.completion_status = "error"
            return state

END = "END"

try:
    from ..models import AgentState, FormField
    from ..agents import (
        FormAnalyzerAgent,
        InformationExtractionAgent,
        AutoFillAgent,
        ValidationAgent,
        QuestionGenerationAgent
    )
except ImportError:
    # Mock classes for testing
    class AgentState:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class FormField:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockAgent:
        def __init__(self):
            pass
    
    FormAnalyzerAgent = MockAgent
    InformationExtractionAgent = MockAgent  
    AutoFillAgent = MockAgent
    ValidationAgent = MockAgent
    QuestionGenerationAgent = MockAgent


class FormFillingWorkflow:
    """Main workflow orchestrator for intelligent form filling"""
    
    def __init__(self):
        self.form_analyzer = FormAnalyzerAgent()
        self.info_extractor = InformationExtractionAgent()
        self.auto_filler = AutoFillAgent()
        self.validator = ValidationAgent()
        self.question_generator = QuestionGenerationAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("analyze_form", self._analyze_form_node)
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("auto_fill", self._auto_fill_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("ask_questions", self._ask_questions_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_form")
        
        # Define workflow edges
        workflow.add_edge("analyze_form", "extract_info")
        workflow.add_edge("extract_info", "auto_fill")
        workflow.add_edge("auto_fill", "validate")
        
        # Add conditional edges for validation results
        workflow.add_conditional_edges(
            "validate",
            self._decide_next_step,
            {
                "complete": END,
                "ask_questions": "ask_questions"
            }
        )
        
        workflow.add_edge("ask_questions", END)
        
        return workflow
    
    def _analyze_form_node(self, state: AgentState) -> AgentState:
        """Node for form analysis"""
        return self.form_analyzer.analyze(state)
    
    def _extract_info_node(self, state: AgentState) -> AgentState:
        """Node for information extraction"""
        return self.info_extractor.extract(state)
    
    def _auto_fill_node(self, state: AgentState) -> AgentState:
        """Node for auto-filling form fields"""
        return self.auto_filler.auto_fill(state)
    
    def _validate_node(self, state: AgentState) -> AgentState:
        """Node for validation"""
        return self.validator.validate(state)
    
    def _ask_questions_node(self, state: AgentState) -> AgentState:
        """Node for generating questions"""
        return self.question_generator.generate_questions(state)
    
    def _decide_next_step(self, state: AgentState) -> Literal["complete", "ask_questions"]:
        """Decide whether to complete or ask questions"""
        # Check if there are validation errors or missing required fields
        has_errors = len(state.validation_errors) > 0
        
        missing_required = any(
            not field.field_value and field.is_required 
            for field in state.form_fields
        )
        
        if has_errors or missing_required:
            return "ask_questions"
        else:
            return "complete"
    
    async def process_form(self, message: str, form_fields: list, user_context: dict = None) -> Dict[str, Any]:
        """
        Process a form filling request through the agentic workflow
        
        Args:
            message: User's message
            form_fields: List of form field dictionaries
            user_context: Optional user context
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Convert form fields to Pydantic models
            parsed_fields = []
            for field_data in form_fields:
                # Handle field label mapping for compatibility
                field_label = field_data.get("fieldLabel", field_data.get("field_label", ""))
                field_type = field_data.get("fieldType", field_data.get("field_type", "text"))
                field_value = field_data.get("fieldValue", field_data.get("field_value", ""))
                data_id = field_data.get("data_id", "")
                
                # Determine if field is required
                is_required = field_label.startswith("*") if field_label else False
                
                form_field = FormField(
                    data_id=data_id,
                    field_label=field_label,
                    field_type=field_type,
                    field_value=field_value,
                    is_required=is_required
                )
                parsed_fields.append(form_field)
            
            # Initialize state
            initial_state = AgentState(
                form_fields=parsed_fields,
                user_message=message,
                user_context=user_context or {},
                current_step="analyze",
                completion_status="in_progress"
            )
            
            # Run the workflow
            result = await self.app.ainvoke(initial_state)
            
            # Prepare response
            response = {
                "status": result.completion_status,
                "filled_fields": {field.data_id: field.field_value for field in result.form_fields if field.field_value},
                "missing_information": result.missing_information,
                "validation_errors": [error.dict() for error in result.validation_errors],
                "questions_for_user": result.questions_for_user,
                "suggestions": result.suggestions,
                "confidence_score": result.confidence_score,
                "workflow_state": result.current_step
            }
            
            return response
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "filled_fields": {},
                "missing_information": [],
                "validation_errors": [],
                "questions_for_user": ["There was an error processing your form. Please try again."],
                "suggestions": [],
                "confidence_score": 0.0,
                "workflow_state": "error"
            }


# Initialize the workflow instance
form_filling_workflow = FormFillingWorkflow()
