#!/usr/bin/env bash

# Start FastAPI in background
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait a bit to ensure backend starts
sleep 5

# Start Streamlit
python -m streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0