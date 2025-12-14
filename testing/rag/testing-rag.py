from google import genai
import os

from src.RAG.rag_engine import RAGEngine
from src.VectorStore.vector_store import get_vectorstore
from src.Embeddings.embedding_provider import get_embeddings

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

def test_rag():
    embeddings = get_embeddings()

    vectordb = get_vectorstore(
        collection_name="test_collection",
        embeddings=embeddings,
        persist_directory="chroma_db"
    )

    rag = RAGEngine(vectordb)

    question = "What is Retrieval Augmented Generation?"

    answer = rag.answer(
        question=question,
        client=client,
        model_name="gemini-2.0-flash"
    )

    print("QUESTION:", question)
    print("=" * 60)
    print("ANSWER:\n", answer)

if __name__ == "__main__":
    test_rag()
