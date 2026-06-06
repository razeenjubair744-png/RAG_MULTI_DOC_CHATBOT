#!/usr/bin/env python3
"""
CLI script to ingest PDFs into ChromaDB.

Usage:
    python ingest.py ./docs/
    python ingest.py ./docs/report.pdf
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingestion.loader import load_pdf, load_pdfs_from_folder
from ingestion.splitter import split_documents
from ingestion.embedder import embed_and_store


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <pdf_file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    start = time.time()

    print(f"\n{'='*50}")
    print(f"RAG Ingestion Pipeline")
    print(f"{'='*50}")
    print(f"Target: {target}\n")

    # Step 1: Load
    if target.is_dir():
        docs = load_pdfs_from_folder(str(target))
    elif target.suffix.lower() == ".pdf":
        docs = load_pdf(str(target))
    else:
        print(f"ERROR: Expected a .pdf file or folder, got: {target}")
        sys.exit(1)

    # Step 2: Split
    chunks = split_documents(docs)

    # Step 3: Embed & store
    stats = embed_and_store(chunks)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  Files processed : {len(stats['unique_files'])}")
    print(f"  Chunks stored   : {stats['chunks_added']}")
    print(f"  Collection      : {stats['collection']}")
    print(f"  Persist dir     : {stats['persist_dir']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
