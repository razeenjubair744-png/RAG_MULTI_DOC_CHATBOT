"""
Embeds document chunks and stores them in ChromaDB.
Also exposes helpers for deleting and listing documents.
"""

import os
from typing import List, Dict, Any

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "rag_documents"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_vectorstore() -> Chroma:
    """Return a persistent ChromaDB vectorstore."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def embed_and_store(chunks: List[Document]) -> Dict[str, Any]:
    """Embed chunks and upsert them into ChromaDB. Returns stats."""
    vectorstore = get_vectorstore()

    # Add documents in batches of 100 to avoid rate limits
    batch_size = 100
    total_added = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        total_added += len(batch)
        print(f"[embedder] Stored {total_added}/{len(chunks)} chunks...")

    unique_files = list({c.metadata.get("source_file", "?") for c in chunks})

    return {
        "chunks_added": total_added,
        "unique_files": unique_files,
        "collection": COLLECTION_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
    }


def delete_document(source_file: str) -> int:
    """Delete all chunks belonging to a source file. Returns count deleted."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.get(where={"source_file": source_file})
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"[embedder] Deleted {len(ids_to_delete)} chunks for '{source_file}'")

    return len(ids_to_delete)


def list_documents() -> List[Dict[str, Any]]:
    """Return a list of unique documents with their chunk counts."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    results = collection.get(include=["metadatas"])
    doc_map: Dict[str, Dict] = {}

    for meta in results["metadatas"]:
        src = meta.get("source_file", "unknown")
        if src not in doc_map:
            doc_map[src] = {
                "source_file": src,
                "total_pages": meta.get("total_pages", "?"),
                "chunk_count": 0,
            }
        doc_map[src]["chunk_count"] += 1

    return list(doc_map.values())
