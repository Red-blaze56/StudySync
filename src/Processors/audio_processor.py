from pathlib import Path
from typing import List
from langchain_core.documents import Document
from faster_whisper import WhisperModel

def get_whisper_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

def load_audio(audio_path: Path, model) -> List[Document]:
    segments, info = model.transcribe(str(audio_path))
    
    text_parts=[]
    for seg in segments:
        if seg.text.strip():
            text_parts.append(seg.text.strip())
    
    full_text = " ".join(text_parts)

    if not full_text.strip():
        return []
    
    return [
        Document(
            page_content=full_text,
            metadata={
                'source_type':"audio",
                "file_name":audio_path.name,
                "language":info.language,
                "duration":info.duration, 
            }
        )
    ]
    

