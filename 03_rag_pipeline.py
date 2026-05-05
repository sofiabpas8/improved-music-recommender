"""
STEP 3 — RAG Recommendation Pipeline + Conversational Explanation

This script runs TWO RAG recommender models (one per embedding type built in
Step 2) and compares their results against the k-NN cosine baseline from
Step 1b.

Architecture
────────────
  User query
      │
      ├─► k-NN Cosine Baseline (01b_baseline_knn.py)
      │
      ├─► RAG Model A  →  ChromaDB A  →  Retriever A  →┐
      │                                                  ├─► LLM (Ollama / API)
      └─► RAG Model B  →  ChromaDB B  →  Retriever B  →┘
                                                         │
                                                         └─► Conversational answer
                                                             (song name + natural-
                                                              language explanation)

Design decision — where does the LLM live?
──────────────────────────────────────────
  The LLM is shared across both RAG models and the baseline. You do NOT need
  a separate LLM for "adding the natural language part"; the same model that
  evaluates retrieved context can also produce the conversational explanation.

  The prompt template below instructs the LLM to:
    1. Recommend a specific song from the retrieved results.
    2. Explain in plain language WHY that song was recommended
       (shared mood, lyrical themes, sonic qualities, etc.)

  This means the embedding model and the LLM are two distinct components:
    · Embedding model → encodes lyrics into vectors for retrieval
    · LLM             → reads retrieved lyrics + user query and writes the
                        explanation in natural language

  You only need one LLM. What changes between Model A and Model B is the
  embedding used for retrieval; the LLM and the prompt stay the same so
  that the comparison is fair.

TODO — configure the LLM and embedding models before running.
"""

import json
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR_A  = "data/chroma_db_model_a"
CHROMA_DIR_B  = "data/chroma_db_model_b"
DATASET_PATH  = "data/eval_dataset.json"
OLLAMA_MODEL  = "mistral"   # ← change to any locally available Ollama model
RESULTS_PATH  = "data/rag_results.json"
TOP_K         = 3           # chunks to retrieve per query

os.makedirs("data", exist_ok=True)

# ── LLM ───────────────────────────────────────────────────────────────────────
# A single LLM is reused for both RAG models and the baseline.
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
# The LLM receives retrieved song lyrics as context and must:
#   (a) answer the multiple-choice question, AND
#   (b) produce a short conversational explanation.
PROMPT_TEMPLATE = """\
You are a music expert and recommender assistant.

You have been asked about the following song:
  Song   : {song}
  Artist : {artist}

Here are lyrics from similar songs retrieved from our catalog:
──────────────────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────

Question: {question}
Options:
  A) {option_a}
  B) {option_b}
  C) {option_c}
  D) {option_d}

Instructions:
  1. On the FIRST line write ONLY the letter of the correct answer (A, B, C or D).
  2. On the SECOND line write a short, friendly explanation (2–4 sentences) of why
     the retrieved songs are similar to '{song}' — mention shared themes, mood, or
     lyrical style. Write as if you are talking directly to the user.

Answer:\
"""

# ── Evaluation helpers ────────────────────────────────────────────────────────

def run_rag_model(name: str, embeddings, chroma_dir: str, dataset: list) -> dict:
    """Run a full RAG evaluation for one embedding model."""
    if embeddings is None:
        print(f"\n⚠  Skipping {name} — embedding model not configured (see TODO above).")
        return {}

    if not os.path.exists(chroma_dir):
        print(f"\n⚠  Skipping {name} — vector store not found at '{chroma_dir}'.")
        print("   Run 02_build_vectorstore.py first.")
        return {}

    print(f"\n{'='*50}")
    print(f"  Running RAG Model: {name}")
    print(f"{'='*50}")

    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    results = []
    correct = 0

    for item in dataset:
        query = f"{item['song']} by {item['artist']}: {item['question']}"

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K,
                "filter": {"title": item["song"].lower()},
            }
        )
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = PROMPT_TEMPLATE.format(
            song=item["song"],
            artist=item["artist"],
            context=context,
            question=item["question"],
            option_a=item["options"]["A"],
            option_b=item["options"]["B"],
            option_c=item["options"]["C"],
            option_d=item["options"]["D"],
        )

        raw = llm.invoke(prompt).strip()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        predicted    = lines[0][0].upper() if lines and lines[0][0].upper() in "ABCD" else "?"
        explanation  = lines[1] if len(lines) > 1 else "(no explanation generated)"
        is_correct   = predicted == item["answer"]
        if is_correct:
            correct += 1

        result = {
            "id":               item["id"],
            "song":             item["song"],
            "question":         item["question"],
            "expected":         item["answer"],
            "predicted":        predicted,
            "correct":          is_correct,
            "explanation":      explanation,
            "retrieved_sources": [d.metadata.get("title") for d in docs],
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Q{item['id']} ({item['song']}): expected={item['answer']}, got={predicted}")
        print(f"       💬 {explanation}\n")

    accuracy = correct / len(dataset) * 100
    print(f"\n  Accuracy ({name}): {correct}/{len(dataset)} = {accuracy:.1f}%")
    return {"model": name, "accuracy": accuracy, "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

all_results = []

result_a = run_rag_model(EMBEDDING_A_NAME, embeddings_a, CHROMA_DIR_A, dataset)
if result_a:
    all_results.append(result_a)

result_b = run_rag_model(EMBEDDING_B_NAME, embeddings_b, CHROMA_DIR_B, dataset)
if result_b:
    all_results.append(result_b)

# ── Summary comparison ────────────────────────────────────────────────────────
if all_results:
    print(f"\n{'='*50}")
    print("  SUMMARY — Model Comparison")
    print(f"{'='*50}")
    for r in all_results:
        print(f"  {r['model']:30s}  →  {r['accuracy']:.1f}% accuracy")
    print()

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to '{RESULTS_PATH}'")
