# Form-Filling Agent with LangGraph

An intelligent agentic AI application built with LangGraph and Azure OpenAI that automatically fills web forms through natural language conversation. This agent uses sophisticated prompt engineering and state management to extract information from user messages, validate data, and guide users through form completion.

## Overview

The form-filling agent transforms the traditional form-filling experience by enabling users to complete complex forms through natural language conversation. The system features:

1. **Intelligent Form Analysis**: Automatically extracts relevant information from user messages and maps it to form fields
2. **Dynamic Field Processing**: Processes individual fields with validation, error handling, and contextual questioning
3. **State-Driven Workflow**: Maintains conversation state and form progress using LangGraph's state management
4. **Azure OpenAI Integration**: Leverages Azure OpenAI for natural language understanding and generation
5. **Extensible Agent Architecture**: Modular design with separate agents for form analysis and field processing

## Architecture

The form-filling agent is built using a modern agentic AI architecture with the following components:

### Core Components

- **LangGraph StateGraph**: Orchestrates the conversation flow and maintains state between interactions
- **Azure OpenAI Integration**: Powers the language understanding and generation capabilities
- **Agent-Based Processing**: Specialized agents handle different aspects of form filling
- **State Management**: Comprehensive state tracking for form fields, conversation history, and user progress
- **Memory Persistence**: Optional Redis-based persistence for conversation continuity

### Agent Architecture

1. **Analyze Form Agent** (`analyze_form_agent.py`): 
   - Extracts information from user messages
   - Maps user input to form field schema
   - Identifies filled and missing fields
   - Integrates with AI search for enhanced information retrieval

2. **Process Field Agent** (`process_field_agent.py`):
   - Handles targeted field-specific processing
   - Validates field inputs and formats
   - Generates contextual follow-up questions
   - Manages field-level error handling

### Workflow Process

1. **Initial Analysis**: User message is processed by the Analyze Form Agent
2. **Field Mapping**: Information is extracted and mapped to form fields
3. **Validation & Processing**: Fields are validated and processed by the Process Field Agent
4. **State Updates**: Form state is updated with filled fields and missing information
5. **Response Generation**: Appropriate response is generated for the user
6. **Routing Logic**: Determines next steps (continue processing, ask for info, or complete form)

## Project Structure

```text
app/formfiller/
├── __init__.py                    # Package initialization and exports
├── README.md                      # This documentation file
├── CHAT_REQUEST_VARS.md          # Developer reference for state variables
├── graph.py                      # Core LangGraph workflow and state management
├── api.py                        # FastAPI endpoints and request/response models
├── llm_client.py                 # Azure OpenAI client configuration
├── agents/                       # Specialized processing agents
│   ├── __init__.py              # Agent module exports
│   ├── analyze_form_agent.py    # Form analysis and information extraction
│   └── process_field_agent.py   # Field-specific processing and validation
└── prompts/                     # LLM prompt templates
    ├── analyze_form_prompt.py   # Main form analysis prompt
    ├── analyze_form_prompt_ReAct.py  # ReAct agent prompt (future use)
    ├── process_field_prompt.py  # Field processing prompt
    └── backup_prompt.py         # Backup/fallback prompts
```

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- Azure OpenAI service account with deployment
- Optional: Redis instance for conversation persistence

### Dependencies

Install the required packages using pip or uv:

```bash
pip install langchain langchain-openai langgraph fastapi uvicorn pydantic python-dotenv redis
```

Or using uv (recommended):

```bash
uv add langchain langchain-openai langgraph fastapi uvicorn pydantic python-dotenv redis
```

### Environment Configuration

Create a `.env` file in your project root with the following Azure OpenAI configuration:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Optional: Redis for conversation persistence
# REDIS_URL=redis://localhost:6379
```

### Local Development Setup

1. **Clone the repository and navigate to the project directory**

2. **Install dependencies** using the commands above

3. **Configure environment variables** as shown in the Environment Configuration section

4. **Optional: Set up Redis for persistence**
   ```bash
   # Using Docker
   docker run -d --name redis -p 6379:6379 redis
   ```

5. **Test the setup**
   ```bash
   python -m app.formfiller.api
   ```

## Usage

### Integration with FastAPI Application

The form-filling agent is integrated into the main FastAPI application. Start the server:

```bash
python -m app.main
```

The agent endpoints are available at `/api/form/` prefix.

### Direct API Usage

You can also run the form-filling API independently:

```bash
python -m app.formfiller.api
```

## API Reference

### Core Endpoints

#### Chat with Form Agent

```http
POST /api/form/chat
```

**Request Body:**
```json
{
  "user_message": "I want to register. My name is John Smith and email is john@example.com",
  "form_fields": [
    {
      "data_id": "name",
      "fieldLabel": "Full Name", 
      "fieldType": "text",
      "fieldValue": null,
      "is_required": true,
      "options": null
    },
    {
      "data_id": "email",
      "fieldLabel": "Email Address",
      "fieldType": "email", 
      "fieldValue": null,
      "is_required": true,
      "options": null
    }
  ],
  "thread_id": "optional-thread-id-for-continuation"
}
```

**Response:**
```json
{
  "thread_id": "uuid-generated-thread-id",
  "response_message": "Thanks! I've captured your name and email. Could you please provide your phone number?",
  "status": "awaiting_info",
  "form_fields": [
    {
      "data_id": "name",
      "fieldLabel": "Full Name",
      "fieldType": "text", 
      "fieldValue": "John Smith",
      "is_required": true,
      "options": null
    }
  ],
  "filled_fields": [
    {
      "data_id": "name",
      "fieldValue": "John Smith"
    },
    {
      "data_id": "email", 
      "fieldValue": "john@example.com"
    }
  ],
  "missing_fields": [
    {
      "data_id": "phone",
      "fieldLabel": "Phone Number",
      "fieldType": "tel",
      "fieldValue": null,
      "is_required": true,
      "options": null,
      "validation_message": "Phone number is required for account registration"
    }
  ]
}
```

### Request/Response Models

#### ChatRequest
- `user_message` (string): The user's natural language input
- `form_fields` (array, optional): Current form field definitions
- `filled_fields` (array, optional): Previously filled field values
- `missing_fields` (array, optional): Fields still requiring input
- `thread_id` (string, optional): Conversation thread identifier
- `conversation_history` (array, optional): Previous message history
- `status` (string, optional): Current conversation status

#### ChatResponse
- `thread_id` (string): Unique conversation identifier
- `response_message` (string): Agent's response to the user
- `status` (string): Current status (`in_progress`, `awaiting_info`, `completed`)
- `form_fields` (array): Updated form field definitions
- `filled_fields` (array): Successfully completed fields
- `missing_fields` (array): Fields requiring user input
- `conversation_history` (array): Complete conversation log

### Status Values

- `in_progress`: Agent is actively processing user input
- `awaiting_info`: Agent needs additional information from user
- `completed`: All required form fields have been filled

## Developer Reference

### State Variables

The form-filling agent uses a comprehensive state management system. For detailed information about all state variables, their types, and usage patterns, see [`CHAT_REQUEST_VARS.md`](./CHAT_REQUEST_VARS.md).

Key state variables include:

- `user_message`: Current user input
- `form_fields`: Complete form field definitions  
- `filled_fields`: Successfully completed fields
- `missing_fields`: Fields requiring user input
- `conversation_history`: Complete chat history
- `thread_id`: Unique conversation identifier
- `status`: Current processing status

### Extending the Agent

#### Custom Prompts

Modify prompts in the `prompts/` directory to customize agent behavior:

- `analyze_form_prompt.py`: Controls information extraction and field mapping
- `process_field_prompt.py`: Handles field validation and follow-up questions
- `analyze_form_prompt_ReAct.py`: ReAct agent implementation (for future use)

#### Custom Agents

Create new agents by following the pattern in `agents/`:

```python
from ..llm_client import llm
from app.formfiller.prompts.your_prompt import your_prompt
from langchain.chains import LLMChain

your_agent_executor = LLMChain(
    llm=llm,
    prompt=your_prompt,
    verbose=True
)
```

#### Advanced Features

1. **Search Integration**: The analyze form agent includes AI search capabilities for enhanced information retrieval
2. **ReAct Agent Support**: Framework ready for ReAct agent implementation
3. **Redis Persistence**: Optional conversation persistence using Redis
4. **Custom Field Types**: Extensible field type system with validation
5. **Multi-language Support**: Configurable for different languages through prompt modification

## Customization and Integration

### Form Field Schema

The agent supports flexible form field definitions:

```json
{
  "data_id": "unique_field_identifier",
  "fieldLabel": "Human-readable label",
  "fieldType": "text|email|tel|number|date|select|checkbox",
  "fieldValue": "current_value_or_null",
  "is_required": true,
  "options": ["array", "of", "options"],
  "validation_message": "Custom validation feedback"
}
```

### Integration Examples

1. **Frontend Integration**: Connect with React/Vue forms using the API endpoints
2. **Webhook Integration**: Process form submissions through webhook handlers
3. **Multi-step Forms**: Handle complex multi-page form workflows
4. **Conditional Logic**: Implement field dependencies and conditional validation
5. **Custom Validation**: Add domain-specific validation rules

### Performance Considerations

- Use Redis for production conversation persistence
- Implement rate limiting for API endpoints
- Consider streaming responses for better user experience
- Cache form field schemas for improved performance
- Monitor Azure OpenAI token usage and costs

## Contributing

When contributing to the form-filling agent:

1. Follow the existing agent pattern for new functionality
2. Update prompts carefully and test thoroughly
3. Maintain backwards compatibility with existing state variables
4. Add comprehensive tests for new features
5. Update this README and `CHAT_REQUEST_VARS.md` for any state changes

## License

This project is licensed under the MIT License - see the [`LICENSE`](./LICENSE) file for details.
