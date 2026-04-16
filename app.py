import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import uvicorn

from src.data_ingestion import process_all_pdfs, split_documents
from src.retriver import RAGRetriever
from src.embedding import EmbeddingManager
from src.vector_store import VectorDB
from src.Llm import get_llm, rag_answer

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded globals
embedding_manager = None
vector_store = None
retriever = None
llm = None

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


def get_llm_instance():
    global llm
    if llm is None:
        print("Loading LLM...")
        llm = get_llm()
    return llm


def get_embedding_manager():
    global embedding_manager
    if embedding_manager is None:
        print("Loading EmbeddingManager...")
        embedding_manager = EmbeddingManager()
    return embedding_manager


def get_vector_store():
    global vector_store
    if vector_store is None:
        print("Loading VectorDB...")
        vector_store = VectorDB()
    return vector_store


def get_retriever():
    global retriever
    if retriever is None:
        print("Loading Retriever...")
        retriever = RAGRetriever(get_vector_store(), get_embedding_manager())
    return retriever


@app.get("/")
def root():
    return {"status": "RAG Chatbot is running"}


@app.get("/health")
def health():
    vs = get_vector_store()
    return {"status": "ok", "chunks_in_store": vs.collection.count()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Saved: {save_path}")

    vs = get_vector_store()
    em = get_embedding_manager()
    get_retriever()  # ensure retriever is initialized

    docs = process_all_pdfs(str(UPLOAD_DIR))
    chunks = split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from PDF.")

    texts = [doc.page_content for doc in chunks]
    embeddings = em.generate_embeddings(texts)
    vs.add_documents(chunks, embeddings)

    print(f"Total chunks now: {vs.collection.count()}")
    return {"message": f"Ingested {len(chunks)} chunks from '{file.filename}'"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    current_llm = get_llm_instance()
    vs = get_vector_store()

    # If no documents uploaded yet, answer with LLM directly
    if vs.collection.count() == 0:
        response = current_llm.invoke(
            f"Answer this question clearly and helpfully:\n\n{request.query}"
        )
        answer = response.content if hasattr(response, "content") else str(response)
        return QueryResponse(answer=answer, sources=["LLM only (no documents uploaded)"])

    ret = get_retriever()
    print(f"Chunks in store: {vs.collection.count()}")

    results = ret.retrieve(
        request.query,
        top_k=request.top_k,
        score_threshold=0.0
    )

    if not results:
        response = current_llm.invoke(
            f"Answer this question clearly and helpfully:\n\n{request.query}"
        )
        answer = response.content if hasattr(response, "content") else str(response)
        return QueryResponse(answer=answer, sources=["LLM only (no relevant context found)"])

    answer = rag_answer(request.query, results, current_llm)
    sources = list({r["metadata"].get("source_file", "unknown") for r in results})
    return QueryResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)