import ollama
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)


def ask_llm(
        prompt: str, 
        system_prompt: str | None = None,
        model: str = "llama3.2",
        temperature: float = 0.0,
        ) -> str:

    """
    Send a prompt to the local LLM and return the response.

    system_prompt sets the LLM's behavior for the whole conversation.
    temperature 0 makes output deterministic (best for RAG); higher values
    make it more creative (better for brainstorming).
    """

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )

    return response["message"]["content"]

if __name__ == "__main__":
    answer = ask_llm(
        prompt="What is retrieval-augmented generation?",
        system_prompt="You are a helpful assistant. Answer in 2 sentences.",
    )
    print(answer)