import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "how do I select rows by position in pandas?"},
)
result = response.json()
print(result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"  - {s['source']} chunk {s['chunk_index']}")