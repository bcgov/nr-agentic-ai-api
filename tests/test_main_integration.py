"""End-to-end integration tests for the NR Agentic AI API."""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from azure.search.documents.models import SearchResult
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nr_agentic_ai_api.main import app  # noqa: E402


class TestEndToEndIntegration:
    """End-to-end integration tests for the complete workflow."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results for testing."""
        return [
            SearchResult(
                score=0.95,
                document={
                    "id": "doc1",
                    "title": "Test Document 1",
                    "content": "This is test content for document 1",
                    "category": "land"
                }
            ),
            SearchResult(
                score=0.87,
                document={
                    "id": "doc2",
                    "title": "Test Document 2",
                    "content": "This is test content for document 2",
                    "category": "water"
                }
            )
        ]

    def test_post_endpoint_integration_with_ai_search_and_workflow(
        self, client, mock_search_results
    ):
        """Test the complete integration workflow."""
        # Mock environment variables
        with patch.dict(os.environ, {
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net/",
            "AZURE_SEARCH_KEY": "test-key",
            "AZURE_SEARCH_INDEX_NAME": "test-index"
        }):
            # Mock the Azure OpenAI LLM
            with patch('nr_agentic_ai_api.main.AzureChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_llm.invoke.side_effect = lambda x: {
                    "output": "I will delegate this request to the Land and Water agents."
                }
                mock_llm_class.return_value = mock_llm

                # Mock the search client
                with patch('nr_agentic_ai_api.main.SearchClient') as mock_search_client_class:
                    mock_search_client = Mock()
                    mock_search_client.search.return_value = mock_search_results
                    mock_search_client_class.return_value = mock_search_client

                    # Mock the Azure credential
                    with patch('nr_agentic_ai_api.main.AzureKeyCredential'):
                        # Test data
                        test_request = {
                            "message": "Find information about land and water resources",
                            "formFields": [
                                {
                                    "data_id": "field1",
                                    "fieldLabel": "Resource Type",
                                    "fieldType": "dropdown",
                                    "fieldValue": "natural_resources"
                                }
                            ],
                            "data": {
                                "priority": "high",
                                "region": "north"
                            },
                            "metadata": {
                                "source": "test_client",
                                "version": "1.0"
                            }
                        }

                        # Make the POST request
                        response = client.post("/api/process", json=test_request)

                        # Assert response status
                        assert response.status_code == 200

                        # Parse response
                        response_data = response.json()

                        # Assert response structure
                        assert response_data["status"] == "success"
                        assert "timestamp" in response_data
                        assert "data" in response_data

                        # Assert processed data structure
                        processed_data = response_data["data"]
                        assert processed_data["received_message"] == test_request["message"]
                        assert processed_data["received_form_fields"] == test_request["formFields"]
                        assert processed_data["received_data"] == test_request["data"]
                        assert processed_data["received_metadata"] == test_request["metadata"]
                        assert "processed_at" in processed_data

                        # Assert workflow outputs
                        assert "orchestrator_output" in processed_data
                        assert "land_output" in processed_data
                        assert "water_output" in processed_data

                        # Verify that the search client was called
                        mock_search_client.search.assert_called()

                        # Verify that the LLM was called for orchestrator
                        mock_llm.invoke.assert_called()

                        # Verify search query was made
                        search_call_args = mock_search_client.search.call_args
                        assert search_call_args[1]["search_text"] == test_request["message"]
                        assert search_call_args[1]["select"] == ["*"]
                        assert search_call_args[1]["top"] == 5


if __name__ == "__main__":
    pytest.main([__file__])
