import requests
import json
from typing import Optional


class OllamaLLM:
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = requests.post(url, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama error {response.status_code}: {response.text}"
            )

        data = response.json()
        return data.get("response", "").strip()
