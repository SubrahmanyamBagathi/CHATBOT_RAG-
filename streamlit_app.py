import streamlit as st
import requests

API_URL = "https://chatbot-rag-cxv9.onrender.com"  # ← your FastAPI backend URL

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot")
st.write("Upload PDFs and ask questions — or just chat without uploading!")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

if st.sidebar.button("🔍 Check Backend"):
    with st.sidebar:
        with st.spinner("Checking..."):
            try:
                res = requests.get(f"{API_URL}/health", timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"✅ Backend alive | Chunks: {data.get('chunks_in_store', 0)}")
                else:
                    st.error(f"Backend returned {res.status_code}")
            except requests.exceptions.Timeout:
                st.warning("⏱️ Backend is waking up (cold start). Wait 30s and try again.")
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tips**\n"
    "- Click **Check Backend** first to wake it up\n"
    "- Upload a PDF before asking doc-specific questions\n"
    "- Without a PDF, the LLM answers from general knowledge"
)

# ── Upload Section ───────────────────────────────────────────────────────────
st.header("📤 Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([1, 4])
    with col1:
        upload_btn = st.button("📨 Upload PDF", type="primary")

    if upload_btn:
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        with st.spinner(f"Processing '{uploaded_file.name}'... this may take a minute."):
            try:
                res = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if res.status_code == 200:
                    st.success(f"✅ {res.json()['message']}")
                else:
                    st.error(f"Upload failed ({res.status_code}): {res.text}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Upload timed out. The backend may be processing — check logs.")
            except Exception as e:
                st.error(f"Upload error: {e}")

# ── Chat Section ─────────────────────────────────────────────────────────────
st.header("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for role, msg, sources in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        with st.chat_message("assistant"):
            st.write(msg)
            if sources and sources != ["LLM only (no documents uploaded)"] and sources != ["LLM only (no relevant context found)"]:
                st.caption(f"📎 Sources: {', '.join(sources)}")
            elif sources:
                st.caption(f"ℹ️ {sources[0]}")

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append(("user", user_input, []))
    st.chat_message("user").write(user_input)

    payload = {"query": user_input, "top_k": 3}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_URL}/query", json=payload, timeout=60)

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                elif res.status_code == 502:
                    answer = "⚠️ Backend is waking up from sleep (cold start). Please wait 30 seconds and try again."
                    sources = []
                elif res.status_code == 503:
                    answer = "⚠️ Backend is temporarily unavailable. Please try again shortly."
                    sources = []
                else:
                    answer = f"❌ Backend error {res.status_code}: {res.text}"
                    sources = []

            except requests.exceptions.Timeout:
                answer = "⏱️ Request timed out. The backend may be cold-starting — please try again in 30 seconds."
                sources = []
            except requests.exceptions.ConnectionError:
                answer = "🔌 Cannot connect to backend. Click **Check Backend** in the sidebar to wake it up."
                sources = []
            except Exception as e:
                answer = f"❌ Unexpected error: {str(e)}"
                sources = []

        st.write(answer)
        if sources and "LLM only" not in sources[0]:
            st.caption(f"📎 Sources: {', '.join(sources)}")
        elif sources:
            st.caption(f"ℹ️ {sources[0]}")

    st.session_state.messages.append(("assistant", answer, sources))