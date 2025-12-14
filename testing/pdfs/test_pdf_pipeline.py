import sys
from pathlib import Path
from src.Processors.pdf_processor import load_pdf

def test_pdf(path):
    print("Runnings")
    docs = load_pdf(path)

    print("=" * 60)
    print("FILE:", path.name)
    print("Documents:", len(docs))

    if not docs:
        print("No docs return")
        return

    total_chars = sum(len(d.page_content) for d in docs)
    print("Total characters:", total_chars)
    print("Metadata sample:", docs[0].metadata)
    print("Preview:")
    print(docs[0].page_content[:300])
    print("=" * 60)


def main():
    print("Running PDF pipeline test...")

    pdf_dir = Path("testing/pdfs")
    pdfs = [p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"]

    print("PDF files found:", pdfs)

    for pdf in pdfs:
        test_pdf(pdf)


if __name__ == "__main__":
    main()