import streamlit as st # type: ignore
import requests

# Internal API (same container)
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot")
st.write("Upload PDFs and ask questions")

# ── Sidebar ─────────────────────────────
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("Check Backend"):
    try:
        res = requests.get(f"{API_URL}/health")
        st.sidebar.success(res.json())
    except:
        st.sidebar.error("Backend not reachable")

# ── Upload Section ──────────────────────
st.header("📤 Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file:
    if st.button("Upload PDF"):
        files = {
            "file": (uploaded_file.name, uploaded_file, "application/pdf")
        }

        with st.spinner("Processing PDF..."):
            try:
                res = requests.post(f"{API_URL}/upload", files=files)

                if res.status_code == 200:
                    st.success(res.json()["message"])
                else:
                    st.error(res.text)

            except Exception as e:
                st.error(str(e))

# ── Chat Section ────────────────────────
st.header("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    payload = {
        "query": user_input,
        "top_k": 3
    }

    with st.spinner("Thinking..."):
        try:
            res = requests.post(f"{API_URL}/query", json=payload)

            if res.status_code == 200:
                answer = res.json()["answer"]
            else:
                answer = "Error from backend"

        except Exception as e:
            answer = str(e)

    st.session_state.messages.append(("bot", answer))

# Display chat
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)