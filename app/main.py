"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.routes.ingest import router as ingest_router
from app.routes.chat import router as chat_router

app = FastAPI(
    title="RAG Chatbot API",
    description="Multi-document Q&A with source citations powered by LangChain + ChromaDB",
    version="1.0.0",
)

# Allow Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "RAG Chatbot API"}
