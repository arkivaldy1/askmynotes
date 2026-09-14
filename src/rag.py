from retrieve import retrieve
from askmynotes.generate import ask_llm

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the user's \
personal notes. You will be given some context from those notes and a question. \

Rules must follow:
1. Answer using ONLY the information in the provided context.
2. If the context does not contain enough information to answer, say \
"I don't have enough information in your notes to answer that."
3. Do not use general knowledge outside the context, even if you know the answer.
4. When you use information from the context, cite the source in square brackets \
like [source: filename, chunk N].
5. Be concise. Answer in 1-4 sentences unless the question needs more depth."""

def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    parts = []
    for chunk in chunks:
        header = f"[source: {chunk['source']}, chunk {chunk['chunk_index']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)

def build_prompt(question: str, context: str) -> str:
    """Construct the final user-facing prompt with context and question."""
    context = format_context(context)
    return f"""Context from your notes:

    {context}

---

Question: {question}

Answer:"""

def rag_query(quesiton: str, top_k: int = 3, model: str = "llama3.2") -> str:
    """
    Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

    Returns a dict with the answer text and the sources used, so callers can
    display or verify them.
    """

    #Step 1: Retrieve
    chunks = retrieve(quesiton, top_k=top_k)

    if not chunks:
        return {
            "answer": "I don't have any relevant information in your ntoes to "
            "answer that question.",
            "sources": [],
            "question": question,
        }

    #Step 2: Build prompt
    user_prompt = build_prompt(quesiton, chunks)

    #Step 3: generate
    answer = ask_llm(
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        model=model,
    )

    return {
        "answer": answer,
        "sources": chunks,
        "question": question,
    }

if __name__ == "__main__":
    import sys

    print("AskMyNotes - type your quesions, or 'quit' to exit\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break

        result = rag_query(question)
        print(f"\nA: {result['answer']}\n")

        if result["sources"]:
            print("Sources:")
            for s in result["sources"]:
                print(f" - {s['source']} (chunk {s['chunk_index']}, "
                    f"distance {s['distance']:.3f})")
        print()
    
