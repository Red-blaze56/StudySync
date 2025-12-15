# StudySync: A Multimodal Retrieval-Augmented Generation (RAG) Based Study Assistant

## Abstract

StudySync is an AI-powered study assistant designed to help students efficiently learn from heterogeneous academic materials such as PDFs, presentation slides, scanned notes, images, audio lectures, and video recordings. Modern students often struggle with fragmented resources spread across multiple formats, making contextual retrieval and revision time-consuming. StudySync addresses this challenge by integrating multimodal content ingestion, semantic search, and Retrieval-Augmented Generation (RAG) into a unified, privacy-first system.

The system preprocesses uploaded materials using specialized pipelines for text extraction, OCR, audio transcription, and video processing. Extracted content is segmented into semantically meaningful chunks and embedded into a vector space using a Sentence Transformer model. These embeddings are stored locally in ChromaDB, enabling fast and accurate semantic retrieval. When a user asks a question, relevant content is retrieved using Max Marginal Relevance (MMR) and passed to a Large Language Model (LLM) to generate grounded, context-aware answers. StudySync supports both online generation (Gemini API) and offline generation (Ollama-based Qwen models). A Streamlit-based interface provides an interactive and intuitive user experience. The modular design ensures extensibility, scalability, and strong data privacy.

## Features

- **Multimodal Content Ingestion**: Supports PDFs, images, audio files (MP3, WAV, M4A), and video files (MP4, MKV, MOV).
- **Advanced Processing Pipelines**:
  - PDF text extraction using PyMuPDF.
  - OCR for scanned documents and images via Gemini Vision API.
  - Audio transcription using Faster-Whisper.
  - Video processing with FFmpeg for audio extraction followed by transcription.
- **Semantic Chunking**: Text is divided into overlapping chunks (size: 600, overlap: 80) to preserve context.
- **Vector Embeddings**: Uses HuggingFace Sentence Transformers (BGE-Large-EN) for embedding generation.
- **Local Vector Storage**: ChromaDB for fast, privacy-preserving semantic search.
- **Retrieval-Augmented Generation (RAG)**: Retrieves top-k (4) relevant chunks using MMR for accurate, context-grounded answers.
- **Dual LLM Support**:
  - Online: Google Gemini (Gemini-2.0-Flash).
  - Offline: Qwen models via Ollama (qwen2.5:7b-instruct).
- **Study Guide Generation**: Automated summarization and structured study guides from ingested materials.
- **User-Friendly Interface**: Streamlit-based web app with tabs for Q&A and study guides.
- **Privacy-First**: All data stored locally; no cloud uploads required.

## Installation

### Prerequisites

- Python 3.12 or higher
- FFmpeg (for video processing)
- Tesseract OCR (for image processing)
- Ollama (for offline LLM support, optional)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd StudySync
   ```

2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```

3. Set up environment variables:
   - For online mode, obtain a Google API key and set `GOOGLE_API_KEY` in a `.env` file.

4. Install Ollama (optional for offline mode):
   - Download and install Ollama from [ollama.ai](https://ollama.ai).
   - Pull the required model: `ollama pull qwen2.5:7b-instruct`

5. Install FFmpeg and Tesseract:
   - FFmpeg: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Tesseract: Install via package manager or from [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

## Usage

1. Run the application:
   ```bash
   streamlit run src/app.py
   ```

2. Open the Streamlit interface in your browser.

3. Select mode: ONLINE or OFFLINE.

4. Upload study materials (PDFs, audio, video files).

5. The system will process and ingest the content into the knowledge base.

6. Use the "Ask Questions" tab to query your materials and get RAG-based answers.

7. Use the "Study Guide" tab to generate summarized study guides.

## System Architecture

The system is modular and consists of:

- **Processors**: Specialized pipelines for different media types (PDF, OCR, Audio, Video).
- **Chunking Module**: Text segmentation using configurable chunk size and overlap.
- **Embedding Provider**: Generates vector embeddings using Sentence Transformers.
- **Vector Store**: ChromaDB for storing and retrieving embeddings.
- **RAG Engine**: Handles retrieval using MMR and context construction.
- **Generator**: Interfaces with LLMs for answer generation (online via Gemini, offline via Ollama).
- **Summarizer**: Creates structured study guides from vector-stored content.
- **Frontend**: Streamlit app for user interaction.

## Tools and Technologies

- **Programming Language**: Python 3.12
- **UI Framework**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers (BGE-Large-EN)
- **OCR**: Gemini Vision API
- **Speech-to-Text**: Faster-Whisper
- **LLMs**: Google Gemini (Gemini-2.0-Flash), Qwen (Ollama)
- **Video Processing**: FFmpeg
- **Other Libraries**: LangChain, PyMuPDF, Pillow, tqdm

## Applications

- Exam preparation and revision
- Summarization of lecture recordings
- Searching handwritten notes
- Concept clarification from personal materials
- Offline-first private study assistant

## Future Scope

- Improved OCR for handwritten notes
- Incremental vector updates
- User-level document filtering
- Citation highlighting in UI
- Mobile-friendly interface
- Cloud synchronization (optional)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
