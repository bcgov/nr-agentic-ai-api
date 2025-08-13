# AI Agent API

A FastAPI-based AI agent API built with LangChain and OpenAI integration.

## Features

- 🚀 **FastAPI** - Modern, fast web framework for building APIs
- 🤖 **LangChain Integration** - Powerful AI/ML framework for building applications
- 🔗 **OpenAI Integration** - GPT models for intelligent responses
- 📝 **Automatic API Documentation** - Interactive docs with Swagger/OpenAPI
- ⚡ **Async Support** - Full asynchronous request handling
- 🔒 **CORS Support** - Cross-Origin Resource Sharing configured
- 🧪 **Testing Ready** - Pytest configuration included
- 📊 **Health Checks** - Built-in health monitoring endpoints

## Project Structure

```
ai-agent/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── chat.py          # Chat endpoints
│   │   │   └── health.py        # Health check endpoints
│   │   └── __init__.py          # API router configuration
│   ├── core/
│   │   ├── config.py            # Application configuration
│   │   └── __init__.py
│   ├── models/
│   │   ├── chat.py              # Pydantic models for chat
│   │   └── __init__.py
│   ├── services/
│   │   ├── ai_service.py        # AI/LangChain service logic
│   │   └── __init__.py
│   └── __init__.py
├── main.py                      # FastAPI application entry point
├── test_main.py                 # Basic tests
├── pyproject.toml               # Project configuration and dependencies
├── env.example                  # Environment variables template
├── start_dev.sh                 # Development server startup script
└── README.md                    # This file
```

## Setup

### 1. Clone and Navigate

```bash
cd /path/to/your/ai-agent
```

### 2. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### 3. Environment Configuration

Copy the environment template and configure your settings:
```bash
cp env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

### 4. Run the Application

Using the development script:
```bash
chmod +x start_dev.sh
./start_dev.sh
```

Or directly with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using Python:
```bash
python main.py
```

## API Endpoints

### Core Endpoints

- **GET /** - Root endpoint with welcome message
- **GET /health** - Basic health check
- **GET /docs** - Interactive API documentation (Swagger)
- **GET /redoc** - Alternative API documentation

### API v1 Endpoints

- **POST /api/v1/chat/** - Chat with the AI agent
- **GET /api/v1/chat/history/{conversation_id}** - Get conversation history
- **GET /api/v1/health/** - Detailed health check
- **GET /api/v1/health/detailed** - System health information

### Example Chat Request

```bash
curl -X POST "http://localhost:8000/api/v1/chat/" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Hello, how can you help me?",
       "conversation_history": [],
       "max_tokens": 1000,
       "temperature": 0.7
     }'
```

## Development

### Running Tests

```bash
pytest test_main.py -v
```

### Code Formatting

Install development dependencies:
```bash
uv sync --extra dev
```

Format code:
```bash
black .
isort .
```

Lint code:
```bash
flake8 .
```

## Configuration

The application can be configured through environment variables or by modifying `app/core/config.py`:

- `PROJECT_NAME` - API project name
- `PROJECT_VERSION` - API version
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: True)
- `ALLOWED_HOSTS` - CORS allowed hosts
- `OPENAI_API_KEY` - OpenAI API key
- `LANGCHAIN_TRACING_V2` - LangChain tracing
- `LANGCHAIN_API_KEY` - LangChain API key

## Docker Support

Create a `Dockerfile` for containerization:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is open source and available under the MIT License.
