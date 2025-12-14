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
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K_RESULTS = 4

# ---------- Models ----------
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "bge-large-en"

#------------System Instructions for LLM--------------
RAG_SYSTEM_INSTRUCTION = """
You are a helpful study assistant.
Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.
"""

SUMMARIZER_SYSTEM_INSTRUCTION = """
You are an expert study-note creator.
Summarize the content clearly and structurally.
Focus on key concepts, definitions, and explanations.
Do not add information that is not present in the source.
"""

STUDY_GUIDE_SYSTEM_INSTRUCTIONS = """
            You are a study assistant creating a study guide for a student.
            Use ONLY the provided context (lecture notes, textbooks, transcripts).
            Create a structured study guide
            Requirements:
            - Organize into sections with headings
            - Summarize key concepts in bullet points
            - Include definitions, key formulas, and examples if present
            - Keep explanations clear and student-friendly
            - Do NOT invent facts not supported by the context
            Now write the study guide:"
    """
