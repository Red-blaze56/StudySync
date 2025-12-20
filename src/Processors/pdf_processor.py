import fitz
from pathlib import Path
from typing import List
from .ocr_processor import ocr_pdf

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path: Path) -> List[Document]:  
    if _has_sufficient_text(pdf_path):
        return _extract_text_direct(pdf_path)
    else:
        return _extract_text_from_ocr(pdf_path)
    
def _extract_text_direct(pdf_path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source_type" : "PDF",
            "file_name" : pdf_path.name,
        })
    
    return docs

def _extract_text_from_ocr(pdf_path: Path) -> List[Document]:
    text = ocr_pdf(pdf_path)

    if not text.strip():
        #text = pytesseract_ocr_pdf(pdf_path)      --to be added as a fail safe in future
        return []
    
    return [
        Document(
            page_content=text,
            metadata={
                "source_type" : "PDF",
                "file_name" : pdf_path.name
            }
        )
    ]

def _has_sufficient_text(pdf_path: Path) ->bool:
    try:
        doc = fitz.open(pdf_path)

        pages_to_check = min(len(doc), 5)
        total_chars = 0

        for i in range(pages_to_check):
            text = doc[i].get_text()
            total_chars += len(text.strip())

        doc.close()

        avg_chars = total_chars / max(pages_to_check, 1)

        return avg_chars >= 120
    except Exception:
        return False
    