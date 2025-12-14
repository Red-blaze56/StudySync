from typing import List
from langchain_core.documents import Document
from google.genai import types


class RAGEngine:
    def __init__(self, vectordb):
        self.retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6}
        )

    def _build_context(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"[Source: {d.metadata.get('file_name', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    def answer(self, question: str, client, model_name: str) -> str:
        docs = self.retriever.invoke(question)

        if not docs:
            return "I don't know. The information is not available in the provided material."

        context = self._build_context(docs)

        user_prompt = f"""
Context:
{context}

Question: {question}

Answer clearly and concisely using ONLY the context above.
"""

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful study assistant for students. "
                "Use ONLY the provided context to answer the question. "
                "If the answer is not in the context, say you don't know."
            ),
            temperature=0.0,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[user_prompt],
            config=config,
        )

        return response.text.strip()
