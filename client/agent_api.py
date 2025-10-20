"""Thin client for interacting with the agent backend API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


class AgentAPIClient:
    """Helper for calling the agent backend endpoints from other services."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        message: str,
        form_state: List[Dict[str, Any]],
        session: Optional[Dict[str, Any]] = None,
        page: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "session": session or {},
            "page": page or {},
            "fields": form_state,
            "history": history or [],
            "prompt": message,
        }
        return self._post("/chat", payload)

    def validate(
        self,
        form_state: List[Dict[str, Any]],
        session: Optional[Dict[str, Any]] = None,
        page: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "session": session or {},
            "page": page or {},
            "fields": form_state,
        }
        return self._post("/validate", payload)

    def apply(
        self,
        form_state: List[Dict[str, Any]],
        session: Optional[Dict[str, Any]] = None,
        page: Optional[Dict[str, Any]] = None,
        confirmed: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "session": session or {},
            "page": page or {},
            "fields": form_state,
            "confirmed": confirmed,
        }
        return self._post("/apply", payload)
