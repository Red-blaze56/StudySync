import streamlit as st
from pathlib import Path
import tempfile
import os

from src.Processors.pdf_processor import load_pdf
from src.Processors.audio_processor import load_audio, get_whisper_model
from src.Processors.video_processor import load_video

from src.Chunking.factory import get_chunker
from src.VectorStore.chroma_store import ChromaStore
from src.RAG.rag_engine import RAGEngine
from src.Summarizer.summarizer import Summarizer
from src.LLM.generator import Generator
from src.config import RAG_SYSTEM_INSTRUCTION

from google import genai
from dotenv import load_dotenv

# --------------------------------------------------
# Setup
# --------------------------------------------------
st.set_page_config(page_title="StudySync", layout="wide")
st.title("📚 StudySync")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --------------------------------------------------
# Sidebar – Mode Selection
# --------------------------------------------------
mode = st.sidebar.radio(
    "Processing Mode",
    ["OFFLINE", "ONLINE"],
)

ollama_model = "qwen2.5:7b-instruct"
gemini_model = "gemini-2.0-flash"

client = None
if mode == "ONLINE":
    client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# Load models once
# --------------------------------------------------
@st.cache_resource
def load_whisper():
    return get_whisper_model()

whisper_model = load_whisper()

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload study materials (PDF / Audio / Video)",
    accept_multiple_files=True,
)

if uploaded_files:
    st.info("Processing files...")

    all_docs = []

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = Path(tmp.name)

        try:
            if suffix == ".pdf":
                docs = load_pdf(tmp_path)

            elif suffix in [".mp3", ".wav", ".m4a"]:
                docs = load_audio(tmp_path, whisper_model)

            elif suffix in [".mp4", ".mkv", ".mov"]:
                docs = load_video(tmp_path, whisper_model)

            else:
                st.warning(f"Unsupported file: {file.name}")
                continue

            all_docs.extend(docs)

        finally:
            tmp_path.unlink(missing_ok=True)

    if not all_docs:
        st.error("No text could be extracted.")
        st.stop()

    # --------------------------------------------------
    # Chunk + Store
    # --------------------------------------------------
    chunker = get_chunker()
    chunks = chunker.chunk(all_docs)

    store = ChromaStore()
    #store.reset() #deleting previous loaded docs in new session
    store.add_documents(chunks)

    st.success(f"Ingested {len(chunks)} chunks into knowledge base.")

    # --------------------------------------------------
    # Tabs: RAG | Summarizer
    # --------------------------------------------------
    tab1, tab2 = st.tabs(["🔍 Ask Questions", "📘 Study Guide"])

    # ==========================
    # RAG TAB
    # ==========================
    with tab1:
        rag = RAGEngine(store)
        generator = Generator()
        question = st.text_input("Ask a question from your materials:")

        retrieved_docs = rag.retrieve(question)

        if not retrieved_docs:
            st.warning("I don't know. No relevant material found.")
            st.stop()

        context = rag.build_context(retrieved_docs)
        prompt = rag.build_prompt(question, context)

        if mode == "ONLINE":
            answer = generator.generate_online(
                prompt=prompt,
                client=client,
                model_name=gemini_model,
                system_instruction=RAG_SYSTEM_INSTRUCTION,
            )
        else:
            answer = generator.generate_offline(
                prompt=prompt,
                model=ollama_model,
                system_instruction=RAG_SYSTEM_INSTRUCTION
            )

        st.write(answer)

    # ==========================
    # SUMMARIZER TAB
    # ==========================
    with tab2:
        summarizer = Summarizer(
            vector_store=store,
            generator=Generator(),
        )

        progress_area = st.empty()

        def update_progress(msg):
            progress_area.info(msg)


        if st.button("Generate Study Guide"):
            with st.spinner("Generating study guide..."):
                guide = summarizer.generate_study_guide(
                    mode=mode,
                    client=client,
                    model_name=gemini_model,
                    ollama_model=ollama_model,
                    progress_callback=update_progress
                )

            st.markdown("### 📘 Study Guide")
            st.write(guide)
