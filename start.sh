#!/usr/bin/env bash

# Start FastAPI in background on same port
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit using Render PORT
python -m streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0