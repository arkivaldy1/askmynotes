from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

_embedding_model = None

def load_text_file(file_path: Path) -> str:
    """Load a text file and return its content as a string."""
    return file_path.read_text(encoding="utf-8")

def load_pdf_file(file_path: Path) -> str:
    """Extracting text from a PDF file, page by page"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def  load_document(file_path: Path) -> str:
    """Route to the right loader based on the file extension"""
    suffix = file_path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return load_text_file(file_path)
    elif suffix == ".pdf":
        return load_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
        Split text into chunks of approximately chunk_size characters, with overlap characters shared between adjecent chunks.

        Overlap prevents ikmportant contrext from being split across chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move start forward by chunk_size minus overlap
    return chunks

def get_embedding_model():
    """Lazy load the embedding model (first run only)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embedding vectors"""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

if __name__ == "__main__":
    # Step 1: Load
    test_file = Path("data/test.md")
    text = load_document(test_file)
    print(f"Loaded {len(text)} characters")

    # Step 2: Chunk
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")

    # Step 3: Embed
    embeddings = embed_texts(chunks)
    print(f"Generated {len(embeddings)} embeddings")

    #Preview
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        print(f"\nChunk {i}: {chunk[:80]}...")
        print(f"Embedding preview: [{emb[0]:.3f}, {emb[1]:.3f}, {emb[2]:.3f}, ...]")

