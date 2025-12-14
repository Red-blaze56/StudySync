from .text_chunker import TextChunker
from .base import BaseChunker


def get_chunker(chunker_type: str = "text") -> BaseChunker:
    if chunker_type == "text":
        return TextChunker()

    raise ValueError(f"Unknown chunker type: {chunker_type}")
