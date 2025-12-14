import requests
from google.genai import types


class Generator:
    def generate_online(self, prompt: str, client, model_name: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful study assistant. "
                "Use ONLY the provided context. "
                "If the answer is not in the context, say you don't know."
            ),
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config,
        )

        return response.text.strip()

    def generate_offline(self, prompt: str, model: str, url: str = "http://localhost:11434/api/generate") -> str:
        system_instruction = (
            "You are a helpful study assistant for students. "
            "Use ONLY the provided context to answer the question. "
            "If the answer is not in the context, say you don't know."
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "system": system_instruction,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        return response.json()["response"].strip()
