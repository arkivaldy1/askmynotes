from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I select rows by position in pandas?",       # 0
    "Use .iloc for integer-based row selection.",        # 1
    "Chocolate cake is my favorite dessert.",            # 2
    "The .iloc method uses integer indexing.",           # 3
]

embeddings = model.encode(sentences).tolist()

print("Embeddings generated for sentences:")
for i, sentence in enumerate(sentences):
    sim = cosine_similarity(embeddings[0], embeddings[i])
    print(f" {sim:.3f} | {sentence}")