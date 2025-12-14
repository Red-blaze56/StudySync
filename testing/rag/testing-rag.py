import os
from pathlib import Path
from google import genai
from dotenv import load_dotenv

from src.RAG.rag_engine import RAGEngine
from src.LLM.generator import Generator
from src.VectorStore.chroma_store import ChromaStore

from src.Processors.pdf_processor import load_pdf
from src.Chunking.factory import get_chunker


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

mode = "OFFLINE" 


def test_rag():
    docs = load_pdf(Path("testing/pdfs/normal.pdf"))
    chunker = get_chunker()
    chunks = chunker.chunk(docs)
    print("loaded and chunked")

 
    store = ChromaStore(collection_name="test_collection")
    store.add_documents(chunks)
    print("added to vector db")

    rag = RAGEngine(store)
    generator = Generator()

    question = "What is Retrieval Augmented Generation?"

    retrieved_docs = rag.retrieve(question)

    if not retrieved_docs:
        print("I don't know")
        return

    # 5. Build context + prompt
    context = rag.build_context(retrieved_docs)
    prompt = rag.build_prompt(question, context)

    # 6. Generate
    if mode == "ONLINE":
        answer = generator.generate_online(
            prompt=prompt,
            client=client,
            model_name="gemini-2.0-flash",
        )
    else:
        answer = generator.generate_offline(
            prompt=prompt,
            model="qwen2.5:7b-instruct",
        )

    print("QUESTION:", question)
    print("=" * 60)
    print("ANSWER:\n", answer)

    # Debug check
    results = store.similarity_search("retrieval augmented generation", k=3)
    print("Retrieved docs:", len(results))


if __name__ == "__main__":
    test_rag()
