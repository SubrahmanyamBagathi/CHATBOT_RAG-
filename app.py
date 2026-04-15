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

# ❗ Lazy-loaded instances (IMPORTANT)
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


@app.get("/")
def root():
    return {"status": "RAG Chatbot is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global embedding_manager, vector_store, retriever

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Saved: {save_path}")

    # 🔥 Lazy initialization
    if embedding_manager is None:
        print("Loading EmbeddingManager...")
        embedding_manager = EmbeddingManager()

    if vector_store is None:
        print("Loading VectorDB...")
        vector_store = VectorDB()

    if retriever is None:
        print("Loading Retriever...")
        retriever = RAGRetriever(vector_store, embedding_manager)

    docs = process_all_pdfs(str(UPLOAD_DIR))
    chunks = split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from PDF.")

    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)

    print(f"Total chunks now: {vector_store.collection.count()}")

    return {"message": f"Ingested {len(chunks)} chunks from '{file.filename}'"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    global retriever, llm

    if retriever is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    # 🔥 Lazy load LLM ONLY when needed
    if llm is None:
        print("Loading LLM...")
        llm = get_llm()

    print(f"Chunks in store: {vector_store.collection.count()}")

    results = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        score_threshold=0.0
    )

    if not results:
        response = llm.invoke(f"Answer this generally: {request.query}")
        answer = response.content if hasattr(response, "content") else str(response)

        return QueryResponse(
            answer=answer,
            sources=["LLM (no context)"]
        )

    answer = rag_answer(request.query, results, llm)
    sources = list({r["metadata"].get("source_file", "unknown") for r in results})

    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    if vector_store is None:
        return {"status": "no data yet"}
    return {"chunks_in_store": vector_store.collection.count()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)