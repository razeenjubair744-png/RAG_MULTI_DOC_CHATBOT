"""
POST /chat         — ask a question, get an answer with sources
DELETE /chat/{sid} — clear conversation memory for a session
"""

import uuid
from typing import List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.chains.rag_chain import ask, clear_memory

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SourceCitation(BaseModel):
    source_file: str
    page_number: Union[int, str]
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    session_id: str


# ── Routes ───────────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question against the ingested documents.
    Pass the same session_id to maintain conversation context.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask(
            question=request.question,
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chain error: {str(e)}")

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        session_id=result["session_id"],
    )


@router.delete("/{session_id}")
def reset_session(session_id: str):
    """Clear chat history for a given session."""
    clear_memory(session_id)
    return {"status": "cleared", "session_id": session_id}
