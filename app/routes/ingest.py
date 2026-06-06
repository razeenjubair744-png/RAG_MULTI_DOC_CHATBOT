"""
POST /ingest  — upload a PDF and add it to the vector store
GET  /docs    — list all ingested documents
DELETE /doc/{filename} — remove a document and its vectors
"""

import os
import tempfile
from pathlib import Path
from typing import List, Union

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from ingestion.embedder import embed_and_store, delete_document, list_documents

router = APIRouter(prefix="/docs", tags=["Documents"])

MAX_FILE_SIZE_MB = 50


# ── Response schemas ─────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    source_file: str
    chunks_added: int
    pages_loaded: int


class DocumentInfo(BaseModel):
    source_file: str
    total_pages: Union[int, str]
    chunk_count: int


class DeleteResponse(BaseModel):
    status: str
    source_file: str
    chunks_deleted: int


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload a PDF and embed it into ChromaDB."""

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read and check size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)",
        )

    # Save to temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        # Override source_file metadata to use the original filename
        for page in pages:
            page.metadata["source_file"] = file.filename

        chunks = split_documents(pages)
        stats = embed_and_store(chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        status="success",
        source_file=file.filename,
        chunks_added=stats["chunks_added"],
        pages_loaded=len(pages),
    )


@router.get("/", response_model=List[DocumentInfo])
def get_documents():
    """List all documents stored in the vector DB."""
    docs = list_documents()
    return [DocumentInfo(**d) for d in docs]


@router.delete("/{filename}", response_model=DeleteResponse)
def remove_document(filename: str):
    """Delete all vectors for a given document filename."""
    deleted = delete_document(filename)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with filename: {filename}",
        )
    return DeleteResponse(
        status="deleted",
        source_file=filename,
        chunks_deleted=deleted,
    )
