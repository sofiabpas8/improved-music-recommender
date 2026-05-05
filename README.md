# 🎵 Music Recommender RAG — Extended Project

An upgraded music recommendation system that compares three approaches:

| Model | Method | Status |
|---|---|---|
| **Baseline** | k-NN · Cosine Similarity on audio features | ✅ Ready (needs CSV dataset) |
| **RAG Model A** | RAG · Embedding type A on lyrics | ⏳ Needs embedding config |
| **RAG Model B** | RAG · Embedding type B on lyrics | ⏳ Needs embedding config |

---

## Changes from the original project

- **k-NN metric**: Euclidean and Manhattan distances removed. Only **Cosine Similarity** is kept — it is the most appropriate metric for high-dimensional feature vectors because it is scale-invariant and measures the angle between vectors rather than raw distance.
- **Dataset**: The Million Song Dataset (MSD) has been removed. A custom **CSV file** will be provided. See `01b_baseline_knn.py` for the expected format.
- **Two RAG models**: The RAG pipeline now runs two embedding models in parallel to compare retrieval quality.
- **Conversational LLM**: The LLM produces a natural-language explanation of each recommendation alongside the answer, using a single shared model (see Architecture below).

---

## Project Structure

```
music-recommender-rag/
├── data/
│   ├── songs_dataset.csv     ← ⚠ TODO: Add your dataset CSV here (lyrics + audio features)
│   ├── chroma_db_model_a/    ← ChromaDB for embedding Model A (auto-generated)
│   ├── chroma_db_model_b/    ← ChromaDB for embedding Model B (auto-generated)
│   ├── eval_dataset.json     ← Multiple-choice evaluation questions
│   └── rag_results.json      ← Output: accuracy + explanations (auto-generated)
├── 01b_baseline_knn.py       ← k-NN Cosine Similarity recommender
├── 02_build_vectorstore.py   ← Embed lyrics into two ChromaDB stores
├── 03_rag_pipeline.py        ← RAG pipeline + evaluation + explanation
├── requirements.txt
└── README.md
```

---

## Architecture

```
User query
    │
    ├─► k-NN Cosine Baseline ──────────────────────────────────► Ranked song list
    │     (audio feature vectors from CSV)
    │
    ├─► RAG Model A → ChromaDB A → Retriever ──┐
    │     (Embedding type A on lyrics)          ├─► Shared LLM ──► Song + explanation
    │                                           │
    └─► RAG Model B → ChromaDB B → Retriever ──┘
          (Embedding type B on lyrics)
```

### LLM placement — one model, two jobs

The LLM is shared across both RAG models. It is **not** embedded inside the
retrieval layer — it sits after it. The flow is:

1. User asks for a recommendation.
2. The retriever finds the most similar song lyrics using the embedding model.
3. The LLM reads the retrieved lyrics and generates:
   - The recommended song name.
   - A conversational explanation of *why* that song is similar (shared mood,
     lyrical themes, sonic qualities, etc.).

You do **not** need a separate LLM just to add natural language. The same
model handles both reasoning and text generation.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull an Ollama model (used as the LLM)
ollama pull mistral
```

---

## TODO before running

1. **Add your dataset**
   Place your CSV file at `data/songs_dataset.csv`.
   Required columns: `title`, `artist`, plus any numeric audio feature columns.

2. **Choose two embedding models**
   Open `02_build_vectorstore.py` and `03_rag_pipeline.py` and fill in the
   `# ← TODO` sections with your chosen embedding models.
   See the comments in those files for examples.

3. *(Optional)* Swap the LLM
   The default LLM is Ollama `mistral`. To use an API-based model (e.g. GPT-4o-mini),
   edit the `llm = ...` line in `03_rag_pipeline.py` and uncomment the relevant
   dependency in `requirements.txt`.

---

## Run order

```bash
# Step 1 — Run k-NN baseline
python 01b_baseline_knn.py

# Step 2 — Build vector stores (both embedding models)
python 02_build_vectorstore.py

# Step 3 — Run RAG pipeline, evaluate, and print explanations
python 03_rag_pipeline.py
```

---

## Evaluation dataset format

Edit `data/eval_dataset.json`:

```json
[
  {
    "id": 1,
    "song": "Bohemian Rhapsody",
    "artist": "Queen",
    "question": "What is the main topic of this song?",
    "options": {
      "A": "A romantic love story",
      "B": "An internal struggle with identity and fate",
      "C": "A political protest",
      "D": "A celebration of friendship"
    },
    "answer": "B"
  }
]
```