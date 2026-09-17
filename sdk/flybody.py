"""Small synchronous Python client for the local FLYLAB Body API v1."""

from __future__ import annotations

from typing import Any, Self
from uuid import uuid4

import httpx


class BodyAPIError(RuntimeError):
    """A protocol rejection; status and response retain the server's machine-readable detail."""

    def __init__(self, status: int, response: Any) -> None:
        self.status = status
        self.response = response
        super().__init__(f"Body API {status}: {response}")


class FlyBody:
    """One controller per brain; call step, advance your world, then submit its reward.

    Transport timeouts retry the exact request once. No request changes neural time
    twice when repeated at the same API boundary. Use one sequential control loop.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        *,
        controller_name: str = "python-game",
        preset: str = "descending",
        timeout: float = 30,
        **configuration: Any,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            trust_env=False,
            headers={"Origin": self.base_url, "X-Flybrain-Local": "1"},
        )
        self.config = {
            "request_id": str(uuid4()),
            "controller_name": controller_name,
            "preset": preset,
            **configuration,
        }
        self.session_id: str | None = None
        self.episode_id: str | None = None
        self.step_index = 0

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(2):
            try:
                response = self.client.post("/api/v1/body" + path, json=payload)
                break
            except httpx.TransportError:
                if attempt:
                    raise
        result: dict[str, Any] = response.json()
        if not response.is_success:
            raise BodyAPIError(response.status_code, result)
        return result

    def _path(self, command: str) -> str:
        if self.session_id is None:
            raise RuntimeError("Call connect() first")
        return f"/sessions/{self.session_id}/{command}"

    def connect(self) -> dict[str, Any]:
        result = self._post("/sessions", self.config)
        self.session_id = result["session_id"]
        self.episode_id = result["episode_id"]
        self.step_index = result["step_index"]
        return result

    def reset(self, seed: int = 0, *, reset_learning: bool = True) -> dict[str, Any]:
        result = self._post(
            self._path("reset"),
            {"request_id": str(uuid4()), "seed": seed, "reset_learning": reset_learning},
        )
        self.episode_id, self.step_index = result["episode_id"], 0
        return result

    def step(self, observation: dict[str, Any], dt_ms: int = 20) -> dict[str, Any]:
        if self.episode_id is None:
            raise RuntimeError("Call reset() before step()")
        result = self._post(
            self._path("step"),
            {
                "episode_id": self.episode_id,
                "step_index": self.step_index + 1,
                "dt_ms": dt_ms,
                "observation": observation,
            },
        )
        self.step_index = result["step_index"]
        return result

    def reward(
        self,
        value: float,
        *,
        components: dict[str, float] | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            self._path("reward"),
            {
                "episode_id": self.episode_id,
                "step_index": self.step_index,
                "value": value,
                "components": components or {},
                "terminated": terminated,
                "truncated": truncated,
            },
        )

    def release(self) -> dict[str, Any] | None:
        if self.session_id is None:
            return None
        result = self._post(self._path("release"))
        self.session_id = self.episode_id = None
        return result

    def close(self) -> None:
        try:
            self.release()
        finally:
            self.client.close()

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
