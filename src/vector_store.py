from pathlib import Path
import chromadb

# Where ChromaDB stores its data on disk
CHROMA_DIR = Path("chroma_db")

# A "collection" is like a table — a named group of vectors
COLLECTION_NAME = "notes"

def get_collection():
    """Get (or create) the ChromaDB collection we'll use for our notes"""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity for vector search
    )

    return collection

def add_chunks(
    chunks: list[str], 
    embeddings: list[list[float]], 
    source_file: str
) -> None:
    """Add chunks and their embeddings to the ChromaDB collection
    
    Each chunk gets a unique id and some metadata (source file and position)
    so we can trace answers back to their origin.
    """
    collection = get_collection()

    #ChromaDB needs unique string IDs for every entry
    ids = [f"{source_file}::chunk_{i}" for i in range((len(chunks)))]

    #Metadata lets us filter and cite sources later
    metadatas = [
        {"source": source_file, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

def collection_stats() -> dict:
    """Quick sanity check - how much is in the store?"""
    collection = get_collection()
    return {
        "name": collection.name,
        "count": collection.count(),
    }

def peek(n: int = 5) -> None:
    """Print the first n entries in the collection."""
    collection = get_collection()
    result = collection.peek(limit=n)
    for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
        print(f"\n--- Entry {i} ---")
        print(f"Source: {meta['source']} (chunk {meta['chunk_index']})")
        print(f"Text: {doc[:150]}...")

def delete_source(source_file: str) -> None:
    """Remove all chunks from a specific source file."""
    collection = get_collection()
    collection.delete(where={"source": source_file})
    print(f"Deleted all chunks from {source_file}")
        

if __name__ == "__main__":
    print(collection_stats())
    peek()

