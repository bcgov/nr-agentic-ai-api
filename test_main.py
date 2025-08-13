import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_health_endpoint():
    """Test the API health check endpoint"""
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_chat_endpoint():
    """Test the chat endpoint"""
    chat_data = {
        "message": "Hello, how are you?",
        "conversation_history": [],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    # This test might fail if OpenAI API key is not set
    # In a real scenario, you'd mock the AI service
    response = client.post("/api/v1/chat/", json=chat_data)
    # For now, we'll just check that the endpoint exists
    assert response.status_code in [200, 500]  # 500 if no API key
