"""Utilities for reading schema JSON definitions from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


class SchemaLoader:
    """Load JSON schema metadata used across validation and retrieval."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def list_schema_files(self) -> Iterable[Path]:
        yield from sorted(self.base_path.glob("*.json"))

    def load_schema(self, page_id: str) -> Dict:
        file_path = self.base_path / f"{page_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        for candidate in self.list_schema_files():
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("page", {}).get("id") == page_id:
                return payload

        raise FileNotFoundError(f"Unknown schema: {page_id}")

    def load_common_types(self) -> Dict:
        file_path = self.base_path / "common-types.json"
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
