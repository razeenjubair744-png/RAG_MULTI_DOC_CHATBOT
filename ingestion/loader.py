"""
Loads PDF files from a file path or directory.
Returns LangChain Document objects with metadata.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF and return its pages as Documents."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Enrich metadata
    for i, page in enumerate(pages):
        page.metadata["source_file"] = Path(file_path).name
        page.metadata["file_path"] = str(file_path)
        page.metadata["page_number"] = i + 1
        page.metadata["total_pages"] = len(pages)

    print(f"[loader] Loaded '{Path(file_path).name}' — {len(pages)} pages")
    return pages


def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """Load all PDFs from a folder recursively."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_files = list(folder.rglob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder_path}")

    all_docs: List[Document] = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(str(pdf_file))
            all_docs.extend(docs)
        except Exception as e:
            print(f"[loader] WARNING: Failed to load {pdf_file.name}: {e}")

    print(f"[loader] Total pages loaded: {len(all_docs)} from {len(pdf_files)} files")
    return all_docs
