from pathlib import Path
from datetime import datetime
from typing import List,Dict

from config import(
    RAW_PDF_DIR,
    RAW_AUDIO_DIR,
    RAW_VIDEO_DIR,
    RAW_IMAGE_DIR,

    VIDEO_FORMATS,
    AUDIO_FORMATS,
    IMAGE_FORMATS,
    PDF_FORMATS 
)

def _get_file_dir(file_suffix: str) -> Path:
    if file_suffix in PDF_FORMATS:
        return RAW_PDF_DIR
    if file_suffix in AUDIO_FORMATS:
        return RAW_AUDIO_DIR
    if file_suffix in VIDEO_FORMATS:
        return RAW_VIDEO_DIR
    if file_suffix in IMAGE_FORMATS:
        return RAW_IMAGE_DIR
    
def ingest_files(uploaded_files)->List[Dict]:
    ingested=[]

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()
        target_dir = _get_file_dir(suffix)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{file.name}"
        saved_path = target_dir/saved_name

        with open(saved_path, "wb") as f:
            f.write(file.getbuffer())

        ingested.append({
            "path": saved_path,
            "type": target_dir.name,   
            "original_name": file.name,
            "ingested_at": timestamp
        })

    return ingested

    

