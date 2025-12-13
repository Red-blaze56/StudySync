from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TEXT_DATA_DIR = DATA_DIR / "text"

RAW_PDF_DIR = RAW_DATA_DIR / "pdf"
RAW_AUDIO_DIR = RAW_DATA_DIR / "audio"
RAW_VIDEO_DIR = RAW_DATA_DIR / "video"
RAW_IMAGE_DIR = RAW_DATA_DIR / "images"

NORMALIZED_TEXT_DIR = TEXT_DATA_DIR / "normalized"

TEMP_DIR = PROJECT_ROOT / "temp"
DB_DIR = PROJECT_ROOT / "chroma_db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for d in [
    RAW_PDF_DIR, RAW_AUDIO_DIR, RAW_VIDEO_DIR, RAW_IMAGE_DIR,
    NORMALIZED_TEXT_DIR, TEMP_DIR, DB_DIR, OUTPUT_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- File formats ----------
VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_FORMATS = {".mp3", ".wav", ".m4a"}
IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}
PDF_FORMATS = {".pdf"}

# ---------- Chunking / RAG ----------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
TOP_K_RESULTS = 6

# ---------- Models ----------
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "bge-large-en"
