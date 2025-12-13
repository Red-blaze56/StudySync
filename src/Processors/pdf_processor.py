import fitz
from pathlib import Path
from typing import List
from Processors.ocr_processor import ocr_pdf

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(pdf_path: Path) -> List[Document]:
    if is_pdf_scanned(pdf_path):
        return _extract_text_direct(pdf_path)
    else:
        return _extract_text_from_ocr(pdf_path)
    
def _extract_text_direct(pdf_path: Path) -> List[Document]:
    loader = PyMuPDFLoader()
    docs = loader.load(pdf_path)

    for doc in docs:
        doc.metadata.update =({"source_type" : "PDF", "file_name" : pdf_path.name})
    
    return docs

def _extract_text_from_ocr(pdf_path: Path) -> List[Document]:
    text = ocr_pdf(pdf_path)

    if not text.strip():
        return []
    
    return Document(
        page_content=text,
        metadata={"source_type" : "PDF", "file_name" : pdf_path.name}
    )

'-------------------------------------------------------------------------------------------------------'

def is_pdf_scanned(pdf_path: Path) -> bool:
    pass

    