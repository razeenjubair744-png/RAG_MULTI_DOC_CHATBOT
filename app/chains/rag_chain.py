"""
Core RAG chain.

Features:
- Hybrid retrieval: ChromaDB (semantic) + BM25 (keyword) via EnsembleRetriever
- MMR to avoid redundant chunks
- ConversationalRetrievalChain for multi-turn memory
- Source document metadata returned with every answer
"""

import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document

from app.chains.prompts import RAG_PROMPT, CONDENSE_QUESTION_PROMPT
from ingestion.embedder import CHROMA_PERSIST_DIR, COLLECTION_NAME

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 5))

# ── In-memory session store  {session_id: memory} ───────────────────────────
_session_memories: Dict[str, ConversationBufferWindowMemory] = {}


def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    if session_id not in _session_memories:
        _session_memories[session_id] = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return _session_memories[session_id]


def clear_memory(session_id: str) -> None:
    _session_memories.pop(session_id, None)


def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _get_all_docs() -> List[Document]:
    """Pull all stored documents for BM25 index."""
    vs = _get_vectorstore()
    results = vs.get()
    docs = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def build_retriever():
    """
    Hybrid retriever: 60% semantic (ChromaDB MMR) + 40% keyword (BM25).
    Falls back to pure semantic if no docs exist yet.
    """
    vectorstore = _get_vectorstore()

    # Semantic retriever with MMR to reduce duplicate chunks
    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3, "lambda_mult": 0.7},
    )

    # BM25 keyword retriever
    all_docs = _get_all_docs()
    if not all_docs:
        return semantic_retriever  # No docs yet, skip BM25

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = TOP_K

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )


def build_rag_chain(session_id: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    retriever = build_retriever()
    memory = get_memory(session_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
        output_key="answer",
        verbose=False,
    )
    return chain


def ask(question: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Run the RAG chain for a question.

    Returns:
        {
            "answer": str,
            "sources": [{"source_file", "page_number", "snippet"}],
            "session_id": str,
        }
    """
    chain = build_rag_chain(session_id)
    result = chain.invoke({"question": question})

    # Format source citations
    sources = []
    seen = set()
    for doc in result.get("source_documents", []):
        meta = doc.metadata
        key = (meta.get("source_file"), meta.get("page_number"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "source_file": meta.get("source_file", "unknown"),
                "page_number": meta.get("page_number", "?"),
                "snippet": doc.page_content[:200].strip(),
            })

    return {
        "answer": result["answer"],
        "sources": sources,
        "session_id": session_id,
    }
