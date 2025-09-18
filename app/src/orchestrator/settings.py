from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class SearchSettings:
    """Configuration for Azure Cognitive Search."""

    endpoint: str | None
    key: str | None
    index: str

    @classmethod
    def from_env(cls) -> "SearchSettings":
        """Load settings from environment variables."""
        load_dotenv(override=True)
        return cls(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            key=os.getenv("AZURE_SEARCH_API_KEY"),
            index=os.getenv("AZURE_SEARCH_INDEX", "wlrs-index"),
        )

    def validate(self) -> None:
        """Ensure endpoint and key are present."""
        if not self.endpoint or not self.key:
            raise RuntimeError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be set to use search"
            )
