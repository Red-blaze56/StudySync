from typing import List, Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.Embeddings.embedding_provider import get_embedding_function
from src.config import DB_DIR


class ChromaStore:
    def __init__(self, collection_name: str = "studysync"):
        self.embedding_fn = get_embedding_function()

        self.store = Chroma(
            collection_name=collection_name,
            persist_directory=str(DB_DIR),
            embedding_function=self.embedding_fn
        )

    def add_documents(self, docs: List[Document]):
        if not docs:
            return
        self.store.add_documents(docs)

    #----------types of searching--------------
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        return self.store.similarity_search(query, k=k)

    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 12) -> List[Document]:
        return self.store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )
    
    #----------------getiing all documents for summarizer \(O_o)/---------------------
    def get_all_documents(self, where: Optional[dict] = None) -> List[Document]:
        data = self.store.get(where=where)
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
    

