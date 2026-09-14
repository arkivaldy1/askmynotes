from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from pathlib import Path

from askmynotes.ingest import load_document, chunk_text, embed_texts
from vector_store import add_chunks, collection_stats
from rag import rag_query

from contextlib import asynccontextmanager
import traceback

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources at startup."""
    from askmynotes.ingest import get_embedding_model
    print("Warming up embedding model...")
    get_embedding_model()
    print("Ready.")
    yield

app = FastAPI(
    title="AskMyNotes API",
    description="RAG-powered Q&A over our personal notes",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev only — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/response models ────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")
    model: str = Field("llama3.2", description="Ollama model to use")

class Source(BaseModel):
    source: str
    chunk_index: int
    distance: float
    text: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources:list[Source]

class IngestResponse(BaseModel):
    filename:str
    chunks_added: int
    total_chunks: int

# ─── Endpoints ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Simple liveness check."""
    return {"status":"ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Submit a question and get a RAG-generated answer with sources
    """
    try:
        result = rag_query(
            question=request.question,
            top_k=request.top_k,
            model=request.model,
        )
    except Exception as e:
        # In real systems you'd log this, not just return it
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {e}")

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document (.md, .txt, or .pdf) and add it to the searchable store.
    """
    #Validatefile type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".md", ".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use .md, .txt, or .pdf."
        )

    # Save the uploaded file to data/
    save_path = DATA_DIR / file.filename
    contents = await file.read()
    save_path.write_bytes(contents)

    # Run the ingestion pipeline
    try:
        text = load_document(save_path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)
        add_chunks(chunks, embeddings, source_file=file.filename)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    stats = collection_stats()
    return IngestResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        total_chunks=stats["count"],
    )

@app.get("/stats")
def stats():
    """Get vector store statistics."""
    return collection_stats()

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")