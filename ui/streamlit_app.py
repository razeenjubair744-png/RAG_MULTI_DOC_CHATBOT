"""
Streamlit chat UI for the RAG Chatbot.

Run with:
    streamlit run ui/streamlit_app.py

Make sure the FastAPI backend is running on port 8000.
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Chatbot")
    st.caption("Multi-document Q&A with source citations")
    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Ingest PDFs", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            for f in uploaded_files:
                with st.spinner(f"Ingesting {f.name}..."):
                    resp = requests.post(
                        f"{API_BASE}/docs/ingest",
                        files={"file": (f.name, f.read(), "application/pdf")},
                    )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"{f.name}: {data['chunks_added']} chunks added")
                    st.session_state.ingested_docs.append(f.name)
                else:
                    st.error(f"{f.name}: {resp.json().get('detail', 'Error')}")

    st.divider()
    st.subheader("Ingested Documents")

    # Fetch live from API
    try:
        doc_resp = requests.get(f"{API_BASE}/docs/", timeout=3)
        docs = doc_resp.json() if doc_resp.status_code == 200 else []
    except Exception:
        docs = []

    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.caption(f"📄 {doc['source_file']} ({doc['chunk_count']} chunks)")
            if col2.button("🗑", key=f"del_{doc['source_file']}", help="Delete"):
                del_resp = requests.delete(
                    f"{API_BASE}/docs/{doc['source_file']}"
                )
                if del_resp.status_code == 200:
                    st.rerun()
    else:
        st.caption("No documents ingested yet.")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        requests.delete(f"{API_BASE}/chat/{st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.divider()
    
    st.subheader("🛠️ Tech Stack")
    st.markdown("""
    - **Frontend:** Streamlit
    - **Backend API:** FastAPI
    - **Models:** OpenAI (`gpt-4o`, `text-embedding-3-small`)
    - **Database:** ChromaDB
    - **Orchestration:** LangChain
    - **Evaluation:** Ragas
    """)

    st.divider()
    st.subheader("📊 Evaluation Metrics")
    st.caption("Latest RAGAS evaluation scores")
    
    import json
    import os
    summary_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faithfulness", f"{summary.get('faithfulness', 0):.2f}")
                st.metric("Context Recall", f"{summary.get('context_recall', 0):.2f}")
            with col2:
                st.metric("Answer Relevance", f"{summary.get('answer_relevancy', 0):.2f}")
                st.metric("Context Precision", f"{summary.get('context_precision', 0):.2f}")
            
            st.caption(f"Evaluated on {summary.get('num_questions', 0)} questions.")
        except Exception as e:
            st.error(f"Could not load metrics: {e}")
    else:
        st.info("No evaluation data found.")
        
    if st.button("Run Evaluation Suite", use_container_width=True):
        with st.spinner("Running RAGAS evaluation... This may take a minute."):
            import subprocess
            import sys
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(__file__))
            try:
                subprocess.run(
                    [sys.executable, "eval/run_eval.py"], 
                    cwd=os.path.dirname(os.path.dirname(__file__)),
                    env=env,
                    check=True
                )
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Evaluation failed with code {e.returncode}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("Ask your documents")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source_file']}** — page {src['page_number']}\n\n"
                        f"> {src['snippet']}..."
                    )

# Chat input
if question := st.chat_input("Ask something about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat/",
                    json={
                        "question": question,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data["sources"]
                else:
                    answer = f"Error: {resp.json().get('detail', 'Unknown error')}"
                    sources = []
            except requests.exceptions.ConnectionError:
                answer = "Cannot connect to the API. Is the FastAPI server running?"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for src in sources:
                    st.markdown(
                        f"**{src['source_file']}** — page {src['page_number']}\n\n"
                        f"> {src['snippet']}..."
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
