"""
Splits LangChain Documents into smaller chunks.
Uses RecursiveCharacterTextSplitter for smart splitting.
"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split documents into chunks.
    Falls back to env vars, then sensible defaults.
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index within the source file
    chunk_counter: dict = {}
    for chunk in chunks:
        src = chunk.metadata.get("source_file", "unknown")
        chunk_counter[src] = chunk_counter.get(src, 0) + 1
        chunk.metadata["chunk_index"] = chunk_counter[src]

    print(
        f"[splitter] {len(documents)} pages → {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
