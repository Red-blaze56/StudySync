import requests
from google.genai import types


class Generator:
    def generate_online(self, prompt: str, client, model_name: str, system_instruction: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=(system_instruction),
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config,
        )

        return response.text.strip()

    def generate_offline(self, prompt: str, system_instruction: str, model: str, url: str = "http://localhost:11434/api/generate") -> str:
        system_instruction = (system_instruction)

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
