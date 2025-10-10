# Form-Filling Agent with LangGraph

An agentic AI application that auto-fills web forms through interactive conversation. This agent can extract information from user messages, identify missing information, and ask specific questions to gather all necessary data.

## Overview

The form-filling agent helps users complete forms through natural language conversation rather than manually filling out individual fields. The agent:

1. Analyzes initial user messages to extract relevant information for form fields
2. Identifies which fields are missing required information
3. Asks the user targeted questions to gather missing information
4. Completes the form when all required information is provided

## Architecture

The agent is built using LangGraph for orchestrating the workflow and LangChain for the language model interactions. The core components are:

- **State Management**: Tracks form fields, filled values, and conversation history
- **Analysis Node**: Extracts information from user messages to populate form fields
- **Field Processing Node**: Handles targeted questions and responses for specific fields
- **Routing Logic**: Determines whether to ask for more information or complete the form

## Project Structure

```
app/formfiller/
├── __init__.py        - Package exports
├── graph.py           - Core LangGraph workflow implementation
├── api.py             - FastAPI endpoints for form-filling agent
├── app_integration.py - Example integration with FastAPI
└── form_filling_agent.ipynb - Jupyter notebook tutorial and demo
```

## Installation

1. Ensure you have Python 3.9+ installed
2. Install required packages:

```bash
pip install langchain langchain-openai langgraph fastapi uvicorn pydantic python-dotenv
```

3. Set up your environment variables:

```
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Usage

### Jupyter Notebook

For exploration and testing, run the `form_filling_agent.ipynb` notebook:

```bash
jupyter notebook app/formfiller/form_filling_agent.ipynb
```

### API Integration

Run the FastAPI application:

```bash
python -m app.formfiller.app_integration
```

## API Endpoints

### Start a Form-Filling Session

```
POST /api/form/start

Request:
{
  "user_message": "I'd like to register. My name is John Smith and my email is john@example.com.",
  "form_fields": [
    {
      "field_id": "name",
      "label": "Full Name",
      "type": "text",
      "required": true
    },
    {
      "field_id": "email",
      "label": "Email Address",
      "type": "email",
      "required": true
    },
    ...
  ]
}

Response:
{
  "thread_id": "uuid-string",
  "response": "Thanks for providing your name and email. Can you also provide your phone number?",
  "status": "awaiting_info",
  "filled_fields": {
    "name": "John Smith",
    "email": "john@example.com"
  },
  "missing_fields": ["phone", "address", ...]
}
```

### Continue a Form-Filling Session

```
POST /api/form/continue

Request:
{
  "thread_id": "uuid-string",
  "user_message": "My phone number is 555-123-4567"
}

Response:
{
  "thread_id": "uuid-string",
  "response": "Thanks! Now can you provide your address?",
  "status": "awaiting_info",
  "filled_fields": {
    "name": "John Smith",
    "email": "john@example.com",
    "phone": "555-123-4567"
  },
  "missing_fields": ["address", ...]
}
```

## Customization

The form-filling agent can be customized for different use cases:

1. **Multiple Form Types**: Add support for different form structures
2. **Validation Rules**: Enhance validation for specific field types
3. **Field Dependencies**: Handle fields that depend on other fields
4. **Custom Prompts**: Tailor the questions based on field context
5. **Integration with Frontend**: Connect with web forms via JS/API

## License

This project is open source and available under the MIT License.



#test for access by Poonam
