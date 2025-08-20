FROM python:3.13-slim-bookworm

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

# Azure App Service expects the app to run on port 8000 by default
EXPOSE 8000

# Set environment variables for Azure App Service 
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONPATH="/app"
# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]