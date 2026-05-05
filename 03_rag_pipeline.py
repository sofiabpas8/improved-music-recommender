"""
STEP 3 — RAG Recommendation Pipeline + Conversational Explanation

This script runs TWO RAG recommender models (one per embedding type built in
Step 2) and compares their results against the k-NN cosine baseline from
Step 1b.

How it works
────────────
  1. The user provides a song title and artist.
  2. The retriever searches the vector store for the most lyrically similar
     songs using the embedding model.
  3. The LLM reads the retrieved lyrics and:
       (a) Picks the single best recommendation from the retrieved songs.
       (b) Explains in natural language why that song is similar to the
           input — shared themes, mood, lyrical style, etc.

Architecture
────────────
  User query (song title + artist)
      │
      ├─► RAG Model A  →  ChromaDB A  →  Retriever A  →┐
      │     (Embedding type A on lyrics)                 ├─► Shared LLM ──► Recommended song
      │                                                  │                  + explanation
      └─► RAG Model B  →  ChromaDB B  →  Retriever B  →┘
            (Embedding type B on lyrics)

Design decision — where does the LLM live?
──────────────────────────────────────────
  The LLM sits after retrieval, not inside the embedding layer. The embedding
  model handles vector search; the LLM reads the retrieved lyrics and writes
  the recommendation and explanation. One LLM is shared across both models so
  the comparison between embeddings is fair.

TODO — configure the LLM and embedding models before running.
"""

import json
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR_A  = "data/chroma_db_model_a"
CHROMA_DIR_B  = "data/chroma_db_model_b"
RESULTS_PATH  = "data/rag_results.json"
OLLAMA_MODEL  = "mistral"   # ← change to any locally available Ollama model
TOP_K         = 1          # number of candidate songs to retrieve (1 for fastest evaluation from users)

os.makedirs("data", exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
# A single LLM is reused for both RAG models so the comparison is fair.
# Using a local Ollama model by default (no API key needed).
# To swap in an API-based model:
#   from langchain_openai import ChatOpenAI
#   llm = ChatOpenAI(model="gpt-4o-mini")
llm = OllamaLLM(model=OLLAMA_MODEL)

# ── Embedding models ──────────────────────────────────────────────────────────
# TODO: Import and instantiate the SAME two embedding models you used in Step 2.
#       The model instances here must match the ones used to build the stores,
#       otherwise ChromaDB will reject the query vectors.

# Model A — replace this block ↓
# Example:
#   from langchain_community.embeddings import HuggingFaceEmbeddings
#   embeddings_a = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings_a = None  # ← TODO
EMBEDDING_A_NAME = "MODEL_A"  # ← TODO: same name you used in 02_build_vectorstore.py

# Model B — replace this block ↓
# Example:
#   from langchain_openai import OpenAIEmbeddings
#   embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_b = None  # ← TODO
EMBEDDING_B_NAME = "MODEL_B"  # ← TODO

# ── Prompt template ───────────────────────────────────────────────────────────
# The retriever finds the most lyrically similar songs.
# The LLM then picks the best one and explains why it matches the input song.
PROMPT_TEMPLATE = """\
You are a music recommendation assistant.

The user is looking for songs similar to:
  Song   : {song}
  Artist : {artist}

Below are lyrics from candidate songs retrieved from our catalog:
──────────────────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

Instructions:
  1. On the FIRST line write ONLY the title and artist of the ONE song from
     the candidates above that is most similar to "{song}" by {artist}.
     Format: Song Title — Artist Name
  2. From the SECOND line onward write a short, friendly explanation (3–5
     sentences) of why you recommend that song — mention shared lyrical themes,
     mood, imagery, or style. Write as if you are talking directly to the user.

Recommendation:\
"""

# ── Recommender ───────────────────────────────────────────────────────────────

def recommend(song: str, artist: str, name: str, embeddings, chroma_dir: str) -> dict:
    """Run a single recommendation for one embedding model."""
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    # Retrieve similar songs, excluding the input song itself
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K,
            "filter": {"title": {"$ne": song.lower()}},
        }
    )
    query = f"{song} by {artist}"
    docs = retriever.invoke(query)

    # Build context block: label each chunk with its song title and artist
    context_parts = []
    for doc in docs:
        label = f"[{doc.metadata.get('title', '?').title()} — {doc.metadata.get('artist', '?').title()}]"
        context_parts.append(f"{label}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(
        song=song,
        artist=artist,
        context=context,
    )

    raw = llm.invoke(prompt).strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    recommended_song = lines[0] if lines else "(no recommendation generated)"
    explanation      = " ".join(lines[1:]) if len(lines) > 1 else "(no explanation generated)"

    return {
        "model":            name,
        "input_song":       song,
        "input_artist":     artist,
        "recommended_song": recommended_song,
        "explanation":      explanation,
        "retrieved_candidates": [
            f"{d.metadata.get('title', '?').title()} — {d.metadata.get('artist', '?').title()}"
            for d in docs
        ],
    }


def run_rag_model(name: str, embeddings, chroma_dir: str, queries: list) -> dict:
    """Run recommendations for a list of input songs with one embedding model."""
    if embeddings is None:
        print(f"\n⚠  Skipping {name} — embedding model not configured (see TODO above).")
        return {}

    if not os.path.exists(chroma_dir):
        print(f"\n⚠  Skipping {name} — vector store not found at '{chroma_dir}'.")
        print("   Run 02_build_vectorstore.py first.")
        return {}

    print(f"\n{'='*55}")
    print(f"  RAG Model: {name}")
    print(f"{'='*55}")

    results = []
    for q in queries:
        song   = q["song"]
        artist = q["artist"]
        print(f"\n  🎵 Input: {song} — {artist}")

        result = recommend(song, artist, name, embeddings, chroma_dir)
        if result:
            print(f"  ✅ Recommended : {result['recommended_song']}")
            print(f"  💬 Explanation : {result['explanation']}")
            results.append(result)

    return {"model": name, "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────
# Define the songs you want recommendations for.
# Each entry needs a "song" title and "artist" name that exist in your dataset.
QUERIES = [
    {"song": "Bohemian Rhapsody", "artist": "Queen"},
    # Add more songs here as needed:
    # {"song": "Blinding Lights", "artist": "The Weeknd"},
]

all_results = []

result_a = run_rag_model(EMBEDDING_A_NAME, embeddings_a, CHROMA_DIR_A, QUERIES)
if result_a:
    all_results.append(result_a)

result_b = run_rag_model(EMBEDDING_B_NAME, embeddings_b, CHROMA_DIR_B, QUERIES)
if result_b:
    all_results.append(result_b)

# ── Save results ──────────────────────────────────────────────────────────────
if all_results:
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to '{RESULTS_PATH}'")