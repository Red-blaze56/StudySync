from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker

chunk_size = 900
chunk_overlap = 120

class TextChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        chunks = self.splitter.split_documents(docs)

        # Add chunk index for traceability
        for i, doc in enumerate(chunks):
            doc.metadata["chunk_index"] = i

        return chunks
