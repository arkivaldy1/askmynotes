from askmynotes.ingest import embed_texts
from vector_store import get_collection

def retrieve(
        query: str, 
        top_k: int = 5,
        max_distance: float = 1.5) -> list[dict]:
    """
    Find the top_k most relevant chunks for a query, filtered by distance.

    max_distance sets a quality floor: chunks farther than this are excluded.
    For cosine distance, 1.5 is generous; tune based on your data.
    """

    # Embed the query using the same model that embedded the chunks
    query_embedding = embed_texts([query])[0]

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    all_results = [
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

    # Filter out chunks that are too far semantically
    return [r for r in all_results if r["distance"] <= max_distance]


if __name__ == "__main__":
    question = "what colour is the sky?"
    results = retrieve(question, top_k=3)
    print(f"Query: {question}\n")
    print(f"Returned {len(results)} chunks after filtering\n")
    for r in results:
        print(f" distance {r['distance']:.3f}: {r['text'][:80]}...")


