from google import genai
from google.genai import types
from pathlib import Path
from config import LLM_MODEL

import os,time

API_KEY = os.getenv('GOOGLE_API_KEY')

def ocr_pdf(file_path: Path)-> str:
    try:
        client=genai.Client(api_key=API_KEY)
        uploaded = client.files.upload(str(file_path))

        while uploaded.state == 'PROCESSING':
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        
        if uploaded.state == "FAILED":
            return ""

        prompt = (
            "Extract all text from this PDF document.\n\n"
            "Instructions:\n"
            "- Process ALL pages\n"
            "- Preserve paragraph structure\n"
            "- Maintain readability\n"
            "- Do not summarize\n"
            "- Output plain text only"
        )

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                types.Part.from_uri(
                    file_uri=uploaded.uri,
                    mime_type=uploaded.mime_type,
                ),
                prompt,
            ],
        )

        return response.text or ""

    except Exception:
        return "" 