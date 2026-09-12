from askmynotes.ingest import embed_texts
from vector_store import get_collection

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Find the top_k most relevant chunks for a query.

    Returns a list of dicts with the chunk text, source metadata,
    and distance score (lower = more similar for cosine distance).
    """

    # Embed the query using the same model that embedded the chunks
    query_embedding = embed_texts([query])[0]

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        }

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

if __name__ == "__main__":
    question = "What is RAG?"
    print(f"Query: {question}\n")
    results = retrieve(question, top_k=3)

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} (distance {result['distance']:.3f}) ---")
        print(f"Source: {result['source']} (chunk {result['chunk_index']})")
        print(f"Text: {result['text'][:200]}...")
        print()


