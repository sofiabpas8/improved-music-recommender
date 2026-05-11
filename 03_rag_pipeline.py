"""
STEP 3 — RAG Recommendation Pipeline with Late Fusion + Conversational Explanation

Runs TWO RAG recommenders (one per embedding model from Step 2) and compares
their results against the k-NN cosine baseline from Step 1b.

Retrieval strategy — late fusion
─────────────────────────────────
  1. ChromaDB retrieves the top CANDIDATE_K songs by text similarity
     (lyrics + metadata embeddings).
  2. Audio cosine similarity is computed separately on the numeric feature
     vectors for each candidate.
  3. A combined score merges both signals:
         score = ALPHA * audio_sim + (1 - ALPHA) * text_sim
  4. Candidates are re-ranked by combined score; the top song is selected.

LLM role — narration only
──────────────────────────
  The LLM receives the full profiles of both the input song and the
  recommended song (metadata, audio features, lyrics) plus the similarity
  scores, and generates a conversational explanation grounded in that data.
  Ranking is handled entirely by the retrieval + fusion step; the LLM
  does not re-rank.

Architecture
────────────
  User query (song title + artist)
      │
      ├─► RAG Model A  →  ChromaDB A (MiniLM)  →  text_sim  ─┐
      │                                                        ├─► late fusion ──► re-rank ──► LLM ──► explanation
      └─► RAG Model B  →  ChromaDB B (mpnet)   →  text_sim  ─┘
                                                   audio_sim ─┘  (shared, model-agnostic)
"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "data/songs_dataset.csv"
CHROMA_DIR_A  = "data/chroma_db_minilm"
CHROMA_DIR_B  = "data/chroma_db_mpnet"
RESULTS_PATH  = "data/rag_results.json"
GROQ_MODEL = "llama-3.1-8b-instant"

CANDIDATE_K   = 20    # songs retrieved from ChromaDB before re-ranking
TOP_K         = 1     # final recommendations passed to the LLM
ALPHA         = 0.5   # weight of audio similarity in combined score (0 = text only, 1 = audio only)

os.makedirs("data", exist_ok=True)

# ── Embedding models (must match the ones used in Step 2) ─────────────────────
embeddings_a     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_A_NAME = "all-MiniLM-L6-v2"

embeddings_b     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
EMBEDDING_B_NAME = "all-mpnet-base-v2"

# ── LLM (shared across both models for a fair comparison) ────────────────────
llm = ChatGroq(model=GROQ_MODEL, api_key=os.environ.get("GROQ_API_KEY"))

# ── Audio features (for late fusion) ─────────────────────────────────────────
AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

df = pd.read_csv(DATASET_PATH)
df = df.drop_duplicates(subset=["track_name", "track_artist"])
df["_title_lower"]  = df["track_name"].str.lower().str.strip()
df["_artist_lower"] = df["track_artist"].str.lower().str.strip()

scaler       = StandardScaler()
audio_matrix = scaler.fit_transform(df[AUDIO_COLS].fillna(0))


def get_audio_vector(title: str, artist: str):
    """Return the normalised audio feature vector for a song, or None if not found."""
    mask = (df["_title_lower"] == title.lower().strip()) & \
           (df["_artist_lower"] == artist.lower().strip())
    if mask.sum() == 0:
        mask = df["_title_lower"] == title.lower().strip()
    if mask.sum() == 0:
        return None
    idx = df[mask].index[0]
    return audio_matrix[df.index.get_loc(idx)].reshape(1, -1)


# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a music recommendation assistant. Your task is to explain why a \
recommended song is a good match for the user's input song.

You have been given the full profiles of both songs, including their lyrics, \
genre, and audio characteristics. Ground your explanation in this data — do \
not invent information.

━━━ INPUT SONG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{input_profile}

━━━ RECOMMENDED SONG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{recommended_profile}

━━━ SIMILARITY SCORES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Audio similarity : {audio_sim:.3f}  (how similar the sonic/musical features are)
  Text similarity  : {text_sim:.3f}  (how similar the lyrics and metadata are)
  Combined score   : {combined:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write a short, friendly explanation (3–5 sentences) of why the recommended \
song is a great match. Mention specific shared elements — lyrical themes, \
mood, genre, or audio qualities — drawing directly from the profiles above.\
"""


def build_song_profile(title: str, artist: str, doc_content: str,
                       audio_vec) -> str:
    """Format a song's full profile for the prompt."""
    lines = [doc_content.strip()]
    if audio_vec is not None:
        # Reconstruct readable audio features from the original (unscaled) row
        mask = (df["_title_lower"] == title.lower().strip())
        if mask.sum() > 0:
            row = df[mask].iloc[0]
            audio_str = "  |  ".join(
                f"{col}: {row[col]:.3f}" for col in AUDIO_COLS if col in df.columns
            )
            lines.append(f"Audio features: {audio_str}")
    return "\n".join(lines)


# ── Recommender ───────────────────────────────────────────────────────────────
def recommend(song: str, artist: str, name: str,
              embeddings, chroma_dir: str) -> dict:
    """Run late-fusion retrieval + LLM explanation for one embedding model."""

    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )

    # ── 1. Retrieve top CANDIDATE_K by text similarity ────────────────────────
    query = f"{song} by {artist}"
    results_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=CANDIDATE_K,
        filter={"title": {"$ne": song.lower()}},
    )

    if not results_with_scores:
        return {}

    # ── 2. Late fusion: combine text similarity with audio similarity ──────────
    input_audio = get_audio_vector(song, artist)

    fused = []
    for doc, text_sim in results_with_scores:
        cand_title  = doc.metadata.get("title", "")
        cand_artist = doc.metadata.get("artist", "")
        cand_audio  = get_audio_vector(cand_title, cand_artist)

        if input_audio is not None and cand_audio is not None:
            audio_sim = float(cosine_similarity(input_audio, cand_audio)[0][0])
            audio_sim = max(0.0, audio_sim)   # cosine can be slightly negative
        else:
            audio_sim = 0.0

        combined = ALPHA * audio_sim + (1 - ALPHA) * text_sim
        fused.append((doc, text_sim, audio_sim, combined))

    # ── 3. Re-rank by combined score ──────────────────────────────────────────
    fused.sort(key=lambda x: x[3], reverse=True)
    top = fused[:TOP_K]

    # ── 4. Look up input song's document for the prompt ───────────────────────
    input_docs = vectorstore.similarity_search(
        query,
        k=1,
        filter={"title": song.lower()},
    )
    input_doc_content = input_docs[0].page_content if input_docs else f"Song: {song}\nArtist: {artist}"
    input_audio_vec   = get_audio_vector(song, artist)
    input_profile     = build_song_profile(song, artist, input_doc_content, input_audio_vec)

    # ── 5. Build prompt and call LLM ──────────────────────────────────────────
    best_doc, best_text_sim, best_audio_sim, best_combined = top[0]
    rec_title   = best_doc.metadata.get("title", "?").title()
    rec_artist  = best_doc.metadata.get("artist", "?").title()
    rec_audio   = get_audio_vector(rec_title, rec_artist)
    rec_profile = build_song_profile(rec_title, rec_artist, best_doc.page_content, rec_audio)

    prompt = PROMPT_TEMPLATE.format(
        input_profile     = input_profile,
        recommended_profile = rec_profile,
        audio_sim         = best_audio_sim,
        text_sim          = best_text_sim,
        combined          = best_combined,
    )

    explanation = llm.invoke(prompt).content.strip()

    return {
        "model":            name,
        "input_song":       song,
        "input_artist":     artist,
        "recommended_song": f"{rec_title} — {rec_artist}",
        "audio_similarity": round(best_audio_sim, 4),
        "text_similarity":  round(best_text_sim, 4),
        "combined_score":   round(best_combined, 4),
        "explanation":      explanation,
    }


def run_rag_model(name: str, embeddings, chroma_dir: str,
                  queries: list) -> dict:
    """Run recommendations for a list of queries with one embedding model."""
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
            print(f"     audio={result['audio_similarity']:.3f}  "
                  f"text={result['text_similarity']:.3f}  "
                  f"combined={result['combined_score']:.3f}")
            print(f"  💬 {result['explanation'][:120]}…")
            results.append(result)

    return {"model": name, "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────
QUERIES = [
    {"song": "Bohemian Rhapsody", "artist": "Queen"},
    # {"song": "Blinding Lights", "artist": "The Weeknd"},
]

all_results = []

result_a = run_rag_model(EMBEDDING_A_NAME, embeddings_a, CHROMA_DIR_A, QUERIES)
if result_a:
    all_results.append(result_a)

result_b = run_rag_model(EMBEDDING_B_NAME, embeddings_b, CHROMA_DIR_B, QUERIES)
if result_b:
    all_results.append(result_b)

if all_results:
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\nResults saved to '{RESULTS_PATH}'")
