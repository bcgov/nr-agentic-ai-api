# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /workspace
COPY backend/ ./backend
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install ./backend

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /install /usr/local
COPY backend/ ./backend/

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
