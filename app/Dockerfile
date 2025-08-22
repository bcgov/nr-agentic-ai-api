FROM python:3.13-slim-bookworm

# Install uv (dependency manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install dependencies from pyproject.toml and uv.lock

# Install dependencies from pyproject.toml and uv.lock
RUN uv sync --frozen --no-cache || true
# Fallback: ensure all dependencies are installed with pip
RUN pip install . --no-cache-dir && pip install --no-cache-dir uvicorn

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set environment variables for Azure App Service
ENV PORT=8000
ENV HOST=0.0.0.0
# Health check endpoint for Azure App Service
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

WORKDIR /

# # Run the FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]