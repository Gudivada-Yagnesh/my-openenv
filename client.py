import requests
from typing import Dict, Any, Optional


class OpenEnvClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id: Optional[str] = None

    def reset(self, task_level: str = "EASY") -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/reset", json={"task_level": task_level})
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return data["observation"]

    def step(self, action: str) -> Dict[str, Any]:
        if not self.session_id:
            raise ValueError("Call reset() first.")
        resp = requests.post(
            f"{self.base_url}/step",
            json={"session_id": self.session_id, "action": {"action_type": action}},
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        if not self.session_id:
            raise ValueError("Call reset() first.")
        resp = requests.get(f"{self.base_url}/state/{self.session_id}")
        resp.raise_for_status()
        return resp.json()
