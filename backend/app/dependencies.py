"""Dependency helpers that provide shared service singletons."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .core.config import Settings, get_settings
from .services.azure_clients import (
    AzureCosmosLogger,
    AzureOpenAIChatClient,
    AzureRedisStore,
    RetrievalAugmentor,
)
from .services.schema_loader import SchemaLoader
from .services.validation import Validator


@lru_cache()
def get_schema_loader() -> SchemaLoader:
    base_path = Path(__file__).resolve().parent / "schemas"
    return SchemaLoader(base_path=base_path)


@lru_cache()
def get_validator() -> Validator:
    return Validator(schema_loader=get_schema_loader())


@lru_cache()
def get_settings_cached() -> Settings:
    return get_settings()


@lru_cache()
def get_chat_client() -> AzureOpenAIChatClient:
    return AzureOpenAIChatClient(get_settings_cached())


@lru_cache()
def get_retrieval_augmentor() -> RetrievalAugmentor:
    return RetrievalAugmentor(get_settings_cached(), get_schema_loader())


@lru_cache()
def get_redis_store() -> AzureRedisStore:
    return AzureRedisStore(get_settings_cached())


@lru_cache()
def get_cosmos_logger() -> AzureCosmosLogger:
    return AzureCosmosLogger(get_settings_cached())
