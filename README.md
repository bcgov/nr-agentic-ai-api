# NR Agentic AI API

An agentic AI API built with FastAPI, LangGraph, and LangChain.

## Quick Start

### Prerequisites
- Python 3.10 or higher
- `uv` package manager installed

### Running the Application

The fastest way to run the application is using `uv` with `uvicorn`:

```bash
uv run uvicorn src.nr_agentic_ai_api.main:app --host 0.0.0.0 --port 8000 --reload
```

This command:
- Uses `uv` to manage dependencies and virtual environment
- Runs `uvicorn` with hot reload enabled
- Serves the FastAPI app on all interfaces (0.0.0.0) on port 8000
- Automatically reloads when code changes are detected

### Alternative Run Methods

#### Option 1: Using the run script
```bash
uv run python run.py
```

#### Option 2: Direct module execution
```bash
uv run python -m src.nr_agentic_ai_api.main
```

#### Option 3: Activate virtual environment first
```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
python run.py
```

## API Endpoints

Once running, the following endpoints are available:

- **Root**: `http://localhost:8000/` - Basic status message
- **Health Check**: `http://localhost:8000/health` - Service health status
- **Main API**: `http://localhost:8000/api/process` - POST endpoint for processing requests
- **Interactive Docs**: `http://localhost:8000/docs` - Swagger UI documentation
- **OpenAPI Schema**: `http://localhost:8000/openapi.json` - OpenAPI specification

## API Usage

### POST /api/process

This is the main endpoint that serves as the backbone for receiving requests.

**Request Body:**
```json
{
  "message": "Your message here",
  "data": {
    "key": "value"
  },
  "metadata": {
    "source": "client"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Request processed successfully",
  "data": {
    "received_message": "Your message here",
    "received_data": {"key": "value"},
    "received_metadata": {"source": "client"},
    "processed_at": "2024-01-01T12:00:00.000000"
  },
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

## Development

The application includes:
- FastAPI for the web framework
- Pydantic for data validation
- Hot reload for development
- Structured error handling
- Comprehensive API documentation

## Dependencies

All dependencies are managed through `uv` and specified in `pyproject.toml`:
- FastAPI
- Uvicorn
- Pydantic
- LangGraph
- LangChain
- Python-dotenv
