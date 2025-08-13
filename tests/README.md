# End-to-End Integration Tests

This directory contains end-to-end integration tests for the NR Agentic AI API.

## Test Overview

The main test file `test_main_integration.py` contains a comprehensive test that verifies:

1. **POST Endpoint Integration**: Tests the `/api/process` endpoint
2. **AI Search Tool Integration**: Verifies the Azure Search integration
3. **Workflow Execution**: Tests the complete orchestrator workflow
4. **Response Validation**: Ensures proper response structure and data

## Running the Tests

### Prerequisites

Make sure you have the development dependencies installed:

```bash
uv sync --dev
```

### Run Tests with pytest

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_main_integration.py -v

# Run with coverage
pytest tests/ --cov=src/nr_agentic_ai_api --cov-report=html
```

### Run Tests with Python

```bash
# From the project root
python -m pytest tests/ -v
```

## Test Structure

The test uses comprehensive mocking to avoid external dependencies:

- **Environment Variables**: Mocked Azure service endpoints
- **Azure OpenAI**: Mocked LLM responses
- **Azure Search**: Mocked search client and results
- **FastAPI Client**: Uses TestClient for endpoint testing

## What the Test Validates

1. **Request Processing**: POST endpoint receives and processes requests correctly
2. **AI Search Integration**: Search tool is called with proper parameters
3. **Workflow Execution**: Orchestrator, Land, and Water agents are invoked
4. **Response Format**: Response contains all expected fields and data
5. **Error Handling**: Proper error responses for failures

## Test Data

The test uses realistic test data including:
- Form fields with various types
- Metadata and additional data
- Search queries for natural resources
- Mock search results with different categories

## Troubleshooting

If tests fail, check:
- All dependencies are installed (`uv sync --dev`)
- Python path includes the `src` directory
- Mock objects are properly configured
- Environment variables are mocked correctly
