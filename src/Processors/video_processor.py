import subprocess
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from .audio_processor import load_audio


def load_video(video_path: Path, model) -> List[Document]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        _extract_audio(video_path, audio_path)
        return load_audio(audio_path, model)

    finally:
        if audio_path.exists():
            audio_path.unlink()


def _extract_audio(video_path: Path, audio_path: Path):
    command = [
        "ffmpeg",                    # <-- NOT hardcoded path
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", "loudnorm",
        "-acodec", "pcm_s16le",
        "-y",
        str(audio_path)
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
