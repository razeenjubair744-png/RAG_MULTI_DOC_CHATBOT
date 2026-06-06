#!/bin/bash

# Start FastAPI backend in the background
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on the PORT provided by Render (default 8501)
export PORT=${PORT:-8501}
streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
