Notes
Conda vs uv. I used UV instead because it is more simpler
Git commands:
echo "# askmynotes" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/arkivaldy1/askmynotes.git
git push -u origin main
Module 1:
The concept: Until 2023, running a decent LLM required serious GPU hardware. Ollama changed that — it's a tool that packages open-source models (Llama, Mistral, Phi, Gemma, etc.) into a simple installer and runs them on consumer hardware, including laptops without a discrete GPU.
Why local matters for this project:
No API costs while you experiment
Your notes never leave your machine (privacy)
You learn what's actually happening rather than treating the LLM as a black box
What is ollama? What are the comparisons of other models
Module 2 — Chunking and embeddings
Chunking
If the user asks a simple question but attached a 200 page PDF textbook, there will be some issues
Context window limits. Even models with 128k+ token windows are slow and expensive when packed full
Signal-to-noise ratio. The LLM does better with focused context than a firehose
Cost. More tokens in = more compute, even locally
So we split documents into smaller pieces (Chunks)
Embeddings
Once you have chunks, you need to find the relevant ones for a given question. Old-school approach: keyword search. If the user asks about "pandas loc vs iloc" and a chunk contains those exact words, match found.
Problem: keyword search misses meaning. "How do I select rows by position?" doesn't share any keywords with "iloc is used for integer-location based indexing" — but they're clearly related.
Embeddings solve this. An embedding is a way of representing meaning as a list of numbers (a vector), such that pieces of text with similar meaning end up close together in that number-space.
Imagine every possible sentence gets placed as a dot on a giant 2D map. Sentences about the same topic cluster together. Sentences about unrelated things sit far apart. When someone asks a question, you convert that to a dot on the same map, then find the nearest neighbors — those are your most relevant chunks.
How does the model know how to place them? It's a neural network trained on huge amounts of text with a specific objective: put semantically similar things close together, and dissimilar things far apart. You'll use a pre-trained one; training your own is a Month 6+ topic.
Vocabulary you'll see everywhere
Chunk — a piece of a document (a paragraph, a page section, N tokens)
Embedding — a vector representation of text meaning
Embedding model — the neural network that turns text into embeddings (different from the LLM that generates answers)
Dimensionality — how many numbers in the vector. 384 and 768 are common. Higher = more nuanced but slower and more storage
Semantic search — searching by meaning instead of keywords, which is what embeddings enable
Why the overlap matters
Imagine a chunk ends mid-sentence: "...use .loc for label-based selection and .iloc for integer-pos". The next chunk starts with "ition selection...". Without overlap, neither chunk has the complete idea. With overlap, both chunks contain the full statement (or at least, one of them does), so retrieval works.
A common rule: overlap = 10-20% of chunk_size. We're using 50/500 = 10%.
A brief note on smarter chunking
We're doing simple character-based chunking. Production systems often use:
Token-based chunking — count tokens (subword units) instead of characters, since that's what LLMs actually see
Recursive chunking — split on paragraphs first, then sentences, then characters as needed
Semantic chunking — split at natural topic boundaries using embeddings

Module 3 - Vector search with ChromaDB

What is a vector database? A database optimized for finding nearest neighbors in high-dimensional space.
A regular database (Postgres, SQLite) is great at "find rows where column X equals Y." A vector database is great at "find the 5 vectors closest to this vector." That's the fundamental difference.
Under the hood, they use algorithms like HNSW (Hierarchical Navigable Small World graphs) or IVF (Inverted File Index) to avoid comparing your query against every stored vector — which would be prohibitively slow past a few thousand documents. These algorithms give up perfect accuracy for massive speed gains, which is a fine trade for semantic search.
Why ChromaDB specifically
Runs locally with no setup — SQLite under the hood, one Python import to use
Persistent — your embeddings stay on disk between sessions
Simple API — you'll learn the core operations in 10 minutes
Handles the embedding step for you if you want — you can hand it raw text and it'll embed for you (we won't do this, but it's an option)

Other options you'll hear about: Pinecone (cloud-hosted, paid), Weaviate (self-hosted, more features), Qdrant (Rust-based, fast, self-hostable), pgvector (Postgres extension for teams already on Postgres), FAISS (Facebook's library, no persistence out of the box, more DIY). All work the same way conceptually. Chroma is the friendliest for learning.

Questions
What does this do?   for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
It pairs items from chunks and embeddings, and it also keeps track of the current position in those collections. The inner zip(chunks, embeddings) creates an iterator of tuples like (chunk_0, emb_0), (chunk_1, emb_1), and so on, by taking the first element from each iterable, then the second, and so on.
So each iteration produces something like (0, (chunk_0, emb_0)), (1, (chunk_1, emb_1)), and so on.