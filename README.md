# Agentic Water Licence Backend

This branch hosts the Python/FastAPI backend that powers the agentic assistant for the British Columbia Water Licence application. It exposes the `/chat`, `/validate`, and `/apply` endpoints that the customer UI (delivered separately) calls to orchestrate the form-filling workflow. Azure OpenAI provides the LLM, Azure AI Search powers retrieval, Redis stores transient page state, and Cosmos DB receives durable audit logs. All resources are assumed to reside in **Canada Central**.

## Repository layout

```text
backend/            FastAPI application code, schemas, services, workers
client/             Thin HTTP client for other services to call the API
docker-compose.yml  Local orchestration of the API + Redis
QA/                 Readiness documentation from the original POC
```

The mocked React widget and demo assets from the reference implementation are intentionally excluded from this branch.

## Prerequisites

* Python 3.11+
* Redis 7 (local container is provided via Compose)
* Azure OpenAI deployments for GPT-4o (primary), GPT-4o-mini (fallback), and embeddings
* Azure AI Search index
* Azure Cosmos DB database + container for audit logs

## Environment variables

Create `backend/.env` (see `backend/.env.example`) with the following settings:

```bash
AZURE_OPENAI_ENDPOINT=<https://your-openai-resource.openai.azure.com>
AZURE_OPENAI_API_KEY=<azure-openai-key>
AZURE_OPENAI_DEPLOYMENT_GPT4O=<gpt4o-deployment-name>
AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI=<optional-gpt4o-mini-deployment>
AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS=<text-embedding-deployment>
AZURE_SEARCH_ENDPOINT=<https://your-search-resource.search.windows.net>
AZURE_SEARCH_INDEX=<search-index-name>
AZURE_SEARCH_API_KEY=<search-admin-key>
REDIS_URL=redis://localhost:6379/0
COSMOS_ENDPOINT=<https://your-cosmos-account.documents.azure.com>
COSMOS_KEY=<cosmos-key>
COSMOS_DB_NAME=<cosmos-database-name>
COSMOS_CONTAINER_NAME=<cosmos-container-name>
REGION=canadacentral
```

All settings are required for production. In development you can leave Azure values blank and the service will fail fast during startup, prompting you to provide them.

## Running the API locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e backend/[dev]
cp backend/.env.example backend/.env  # fill in the values above
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternatively run the API and Redis via Docker Compose:

```bash
docker compose up --build
```

## Endpoints

Each endpoint accepts/returns JSON with the form state schema (`id`, `label`, `type`, `required`, `value`, `visibleWhen`, `helpRef`). Example calls:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "session": {"threadId": "demo"},
        "page": {"id": "step-3", "title": "Water Source"},
        "fields": [
          {"id": "source_type", "value": "surface"},
          {"id": "stream_name", "value": "Koksilah River"}
        ],
        "prompt": "What do I need for this page?"
      }'

curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"session": {"threadId": "demo"}, "page": {"id": "step-4"}, "fields": []}'

curl -X POST http://localhost:8000/apply \
  -H "Content-Type: application/json" \
  -d '{"session": {"threadId": "demo"}, "page": {"id": "step-4"}, "fields": [], "confirmed": false}'
```

## Retrieval seeding

The worker ingests schema metadata and provincial guidance into Azure AI Search. It requires the same Azure environment variables used by the API.

```bash
python backend/workers/seed_index.py --dry-run       # preview payloads
python backend/workers/seed_index.py --recreate      # rebuild index and upload
```

## Testing

```bash
pip install -e backend/[dev]
pytest
```

Unit tests cover the conditional visibility engine and validation rules (groundwater well tag, irrigation constraints, etc.).

## Agent API client

Other services can call the backend via `client/agent_api.py`:

```python
from client.agent_api import AgentAPIClient

client = AgentAPIClient(base_url="https://api.example.com")
response = client.validate(form_state=[{"id": "purpose", "value": "irrigation"}], page={"id": "step-4"})
```

## Constraints & notes

* No POSSE scraping is performed by the backend; production UI integrations are strictly client-side.
* Azure resources are assumed to live in Canada Central.
* Redis is used for ephemeral session state; Cosmos DB holds audit trails only—no PII beyond the submitted form payload is persisted.
* The mocked frontend from the reference implementation is intentionally omitted from this branch.
