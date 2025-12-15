from typing import List
from langchain_core.documents import Document
from google.genai import types

class RAGEngine:
    def __init__(self, chroma_store):
        self.store = chroma_store

    def build_context(self, docs, max_chars: int = 6000) -> str:
        context = ""
        for i, doc in enumerate(docs):
            if len(context) >= max_chars:
                break
            context += f"[Chunk {i+1}]\n{doc.page_content}\n\n"
        return context.strip()

        
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.store.mmr_search(query=query, k=k, fetch_k=12)

    def build_prompt(self, question: str, context: str) -> str:
        return f"""
Context:
{context}

Question:
{question}

Answer clearly and concisely using ONLY the context above.
If the answer is not in the context, say you don't know.
""".strip()



