import fitz
from pathlib import Path
from typing import List
from Processors.ocr_processor import ocr_pdf

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path: Path) -> List[Document]:
    docs = _extract_text_direct
    
    if not _has_sufficient_text(pdf_path):
        return _extract_text_from_ocr
    
    return docs
    
def _extract_text_direct(pdf_path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load(pdf_path)

    for doc in docs:
        doc.metadata.update =({
            "source_type" : "PDF", "file_name" : pdf_path.name
        })
    
    return docs

def _extract_text_from_ocr(pdf_path: Path) -> List[Document]:
    text = ocr_pdf(pdf_path)

    if not text.strip():
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

def _has_sufficient_text(docs: List[Document]) ->bool:
    for d in docs:
        total = sum(len(d.page_content.strip()))
        avg_char_per_page=total/max(len(docs),1)

    if total>=200 and avg_char_per_page>=40:
        return True
    else:
        return False

    