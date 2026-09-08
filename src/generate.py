import ollama

def ask_llm(prompt: str, model: str = "llama3.2") -> str:

    """
        Send a prompt to the LLM and return the response.
    """

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]

if __name__ == "__main__":
    prompt = "What is retrieval-augmented generation? Answer in 2 sentences."
    response = ask_llm(prompt)
    print(response)