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

# ── Single shared instances ───────────────────────────────────────────────────
embedding_manager = EmbeddingManager()
vector_store      = VectorDB()
retriever         = RAGRetriever(vector_store, embedding_manager)
llm               = get_llm()

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
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"Saved: {save_path}")

    # ✅ Use the SAME vector_store instance
    docs   = process_all_pdfs(str(UPLOAD_DIR))
    chunks = split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=500, detail="No text extracted from PDF.")

    texts      = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)   # ← same instance

    print(f"Total chunks now: {vector_store.collection.count()}")  # ← confirm
    return {"message": f"Ingested {len(chunks)} chunks from '{file.filename}'"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    print(f"Chunks in store at query time: {vector_store.collection.count()}")  # ← confirm
    results = retriever.retrieve(request.query, top_k=request.top_k, score_threshold=0.0)

    if not results:
         response = llm.invoke(f"Answer this generally: {request.query}")

    # 🔥 Extract actual text from LLM response
         answer = response.content if hasattr(response, "content") else str(response)

         return QueryResponse(
          answer=answer,
          sources=["LLM (no context)"]
        )

    answer  = rag_answer(request.query, results, llm)
    sources = list({r["metadata"].get("source_file", "unknown") for r in results})
    return QueryResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"chunks_in_store": vector_store.collection.count()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)