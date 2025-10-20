"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pathlib import Path

from pydantic import AnyHttpUrl, BaseSettings, Field, validator


class Settings(BaseSettings):
    """Runtime settings for the agent backend."""

    azure_openai_endpoint: AnyHttpUrl = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment_gpt4o: str = Field(..., alias="AZURE_OPENAI_DEPLOYMENT_GPT4O")
    azure_openai_deployment_gpt4o_mini: Optional[str] = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI"
    )
    azure_openai_deployment_embeddings: Optional[str] = Field(
        default=None, alias="AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS"
    )

    azure_search_endpoint: AnyHttpUrl = Field(..., alias="AZURE_SEARCH_ENDPOINT")
    azure_search_index: str = Field(..., alias="AZURE_SEARCH_INDEX")
    azure_search_api_key: Optional[str] = Field(default=None, alias="AZURE_SEARCH_API_KEY")

    redis_url: str = Field(..., alias="REDIS_URL")

    cosmos_endpoint: AnyHttpUrl = Field(..., alias="COSMOS_ENDPOINT")
    cosmos_key: str = Field(..., alias="COSMOS_KEY")
    cosmos_db_name: str = Field(..., alias="COSMOS_DB_NAME")
    cosmos_container_name: str = Field(..., alias="COSMOS_CONTAINER_NAME")

    region: str = Field("canadacentral", alias="REGION")

    class Config:
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        case_sensitive = True

    @validator("region")
    def _normalise_region(cls, value: str) -> str:
        return value.lower()


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance, failing fast if invalid."""

    return Settings()
