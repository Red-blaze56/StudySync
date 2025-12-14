from src.Chunking.factory import get_chunker
from src.Processors.pdf_processor import load_pdf
from pathlib import Path

def test_chunking():
    docs = load_pdf(Path("testing/pdfs/normal.pdf"))
    chunker = get_chunker()
    chunks = chunker.chunk(docs)

    print("Original docs:", len(docs))
    print("Chunks:", len(chunks))
    print("Sample chunk:\n", chunks[0].page_content[:500])

if __name__ == "__main__":
    test_chunking()
