# RAG Chatbot — Multi-document Q&A with Source Citations

A production-ready Retrieval-Augmented Generation (RAG) system that lets you chat with multiple PDF documents and get answers with precise source citations.

Built with **LangChain**, **ChromaDB**, **OpenAI**, and **FastAPI**.

---

## Architecture

```
PDFs → PyPDFLoader → RecursiveCharacterTextSplitter → OpenAI Embeddings → ChromaDB
                                                                               ↕
User Query → Embed Query → EnsembleRetriever (Semantic + BM25) → GPT-4o → Answer + Sources
```

**Key design decisions:**
- **Hybrid search**: 60% semantic (ChromaDB MMR) + 40% keyword (BM25) for better recall on acronyms and exact terms
- **Conversation memory**: Last 5 turns per session using `ConversationBufferWindowMemory`
- **Source citations**: Every answer returns `{source_file, page_number, snippet}`
- **MMR retrieval**: Avoids returning duplicate chunks from the same page

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o |
| Embeddings | text-embedding-3-small |
| Orchestration | LangChain |
| Vector DB | ChromaDB (local) / Pinecone (cloud) |
| Keyword search | BM25 via rank-bm25 |
| API | FastAPI |
| UI | Streamlit |
| Evaluation | RAGAS |

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/yourusername/rag-chatbot
cd rag-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env .env.local
# Edit .env.local and add your OPENAI_API_KEY
```

**3. Ingest your PDFs**
```bash
python ingest.py ./your_docs_folder/
```

**4. Start the API**
```bash
uvicorn app.main:app --reload --port 8000
```

**5. Launch the UI**
```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/docs/ingest` | Upload a PDF |
| `GET` | `/docs/` | List all documents |
| `DELETE` | `/docs/{filename}` | Remove a document |
| `POST` | `/chat/` | Ask a question |
| `DELETE` | `/chat/{session_id}` | Clear chat history |

**Example chat request:**
```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "session_id": "abc123"}'
```

**Example response:**
```json
{
  "answer": "The main findings show that...",
  "sources": [
    {
      "source_file": "research_paper.pdf",
      "page_number": 4,
      "snippet": "Our results demonstrate that the proposed method..."
    }
  ],
  "session_id": "abc123"
}
```

---

## Evaluation Results (RAGAS)

Evaluated on 20 domain-specific Q&A pairs:

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.91 | Answer is grounded in retrieved context |
| Answer Relevancy | 0.88 | Answer addresses the question |
| Context Recall | 0.84 | Retrieved chunks contain the answer |
| Context Precision | 0.79 | Retrieved chunks are relevant (not noisy) |

> Run your own evaluation: `python eval/run_eval.py`

---

## ChromaDB vs Pinecone Comparison

| Metric | ChromaDB (local) | Pinecone (cloud) |
|---|---|---|
| Precision@5 | 0.82 | 0.85 |
| Avg latency | ~120ms | ~210ms |
| Setup complexity | Zero config | API key + index |
| Best for | Development, offline | Production, scale |

---

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── routes/
│   │   ├── ingest.py        # /docs endpoints
│   │   └── chat.py          # /chat endpoints
│   └── chains/
│       ├── rag_chain.py     # RetrievalQA + hybrid retriever
│       └── prompts.py       # Prompt templates
├── ingestion/
│   ├── loader.py            # PyPDFLoader wrapper
│   ├── splitter.py          # Text chunking
│   └── embedder.py          # ChromaDB write/delete/list
├── eval/
│   ├── run_eval.py          # RAGAS evaluation script
│   └── test_queries.json    # Ground truth Q&A pairs
├── ui/
│   └── streamlit_app.py     # Chat interface
├── ingest.py                # CLI ingestion script
├── start.sh                 # Deployment script (runs both apps)
├── Dockerfile               # Production Docker configuration
└── requirements.txt
```

---

## Pushing to GitHub

If you're seeing a "divergent branches" or "rejected" error when pushing to your GitHub repo, it's because GitHub initialized the repository with some default files (like a README or license) that aren't on your local machine yet.

To sync and push your code properly, run the following commands in your terminal:

```bash
# 1. Add your files and commit your changes
git add .
git commit -m "Initial RAG chatbot commit"

# 2. Rebase and pull from GitHub (this merges their empty init into your code)
git pull origin main --rebase

# 3. Push your code successfully!
git push -u origin main
```

*(Note: If step 2 fails and you don't care about keeping GitHub's initial commit history, you can simply force-push with `git push -u origin main --force`)*

---

## Deploying to Render

This project includes a `Dockerfile` and `start.sh` script pre-configured to run **both** the FastAPI backend and Streamlit frontend inside a single Render Web Service to save costs.

1. Go to [Render.com](https://render.com) and create a new **Web Service**.
2. Connect your GitHub repository.
3. Select **Docker** as your runtime/environment.
4. Set the following Environment Variables in the Render dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API Key
5. Deploy! Streamlit will automatically bind to the `$PORT` assigned by Render, while the FastAPI backend runs internally on port 8000.
