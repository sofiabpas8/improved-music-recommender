"""
recommender.py — Unified inference module for the music recommender arena.

Usage (e.g. from a Streamlit app):

    from recommender import load_models, recommend

    load_models()   # call once at startup — loads all indexes into memory

    results = recommend("Yesterday", "The Beatles")

Results format:
    {
        "knn": {
            "recommended_song":   "...",
            "recommended_artist": "...",
            "audio_similarity":   0.87,
            "text_similarity":    0.74,
            "combined_score":     0.80,
        },
        "retrieval_minilm": {
            "recommended_song":   "...",
            "recommended_artist": "...",
            "audio_similarity":   0.87,
            "text_similarity":    0.74,
            "combined_score":     0.80,
            "explanation":        "...",
        },
        "retrieval_mpnet": { ... },
    }

Raises ValueError if the song is not found in the dataset.
Raises RuntimeError if load_models() has not been called first.
"""

import numpy as np
import pandas as pd
import scipy.sparse
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Config ────────────────────────────────────────────────────────────────────
CLEAN_PATH   = "data/songs_clean.csv"
CHROMA_DIR_A = "data/chroma_db_minilm"
CHROMA_DIR_B = "data/chroma_db_mpnet"
GROQ_MODEL = "llama-3.1-8b-instant"
CANDIDATE_K  = 20
TOP_K        = 1
ALPHA        = 0.5

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

# ── Module-level state (populated by load_models) ─────────────────────────────
_df           = None
_audio_matrix = None
_scaler       = None
_tfidf_matrix = None
_tfidf        = None
_chroma_a     = None
_chroma_b     = None
_embeddings_a = None
_embeddings_b = None
_llm          = None
_loaded       = False


def load_models(use_llm: bool = True):
    """Load all indexes and models into memory. Call once at app startup.

    Parameters
    ----------
    use_llm : set to False to skip Ollama (retrieval + scores work, no explanations)
    """
    global _df, _audio_matrix, _scaler, _tfidf_matrix, _tfidf
    global _chroma_a, _chroma_b, _embeddings_a, _embeddings_b, _llm, _loaded

    print("Loading dataset...")
    _df = pd.read_csv(CLEAN_PATH)
    _df["_title_lower"]  = _df["track_name"].str.lower().str.strip()
    _df["_artist_lower"] = _df["track_artist"].str.lower().str.strip()

    print("Loading audio index...")
    _audio_matrix = np.load("data/audio_matrix.npy")
    _scaler       = joblib.load("data/audio_scaler.joblib")

    print("Loading TF-IDF index...")
    _tfidf_matrix = scipy.sparse.load_npz("data/tfidf_matrix.npz")
    _tfidf        = joblib.load("data/tfidf_vectorizer.joblib")

    print("Loading embedding models...")
    _embeddings_a = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _embeddings_b = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("Loading ChromaDB stores...")
    _chroma_a = Chroma(persist_directory=CHROMA_DIR_A, embedding_function=_embeddings_a)
    _chroma_b = Chroma(persist_directory=CHROMA_DIR_B, embedding_function=_embeddings_b)

    if use_llm:
        print("Loading LLM...")
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        _llm = ChatGroq(model=GROQ_MODEL, api_key=api_key)
    else:
        print("Skipping LLM (--no-llm mode). Retrieval explanations will be unavailable.")

    _loaded = True
    print("✓ All models loaded.\n")


def _check_loaded():
    if not _loaded:
        load_models()


def _lookup_song(title: str, artist: str):
    """Return the dataframe row index for a song, or raise ValueError."""
    mask = _df["_title_lower"] == title.lower().strip()
    if artist:
        mask &= _df["_artist_lower"] == artist.lower().strip()
    if mask.sum() == 0:
        mask = _df["_title_lower"] == title.lower().strip()
    if mask.sum() == 0:
        raise ValueError(f"'{title}' by '{artist}' not found in the dataset.")
    return _df[mask].index[0]


def _get_audio_vec(title: str, artist: str = ""):
    """Return the pre-scaled audio vector for a song, or None."""
    try:
        idx = _lookup_song(title, artist)
        pos = _df.index.get_loc(idx)
        return _audio_matrix[pos].reshape(1, -1)
    except ValueError:
        return None


# ── k-NN recommendation ───────────────────────────────────────────────────────
def _knn_recommend(song: str, artist: str) -> dict:
    target_idx  = _lookup_song(song, artist)
    target_pos  = _df.index.get_loc(target_idx)

    audio_sims = cosine_similarity(
        _audio_matrix[target_pos].reshape(1, -1), _audio_matrix
    )[0]
    text_sims = linear_kernel(
        _tfidf_matrix[target_pos], _tfidf_matrix
    )[0]

    results = _df[["track_name", "track_artist"]].copy()
    results["audio_similarity"] = audio_sims
    results["text_similarity"]  = text_sims
    results = results.drop(index=target_idx)

    a_min, a_max = results["audio_similarity"].min(), results["audio_similarity"].max()
    t_min, t_max = results["text_similarity"].min(),  results["text_similarity"].max()
    norm_audio = (results["audio_similarity"] - a_min) / (a_max - a_min + 1e-9)
    norm_text  = (results["text_similarity"]  - t_min) / (t_max  - t_min  + 1e-9)
    results["combined_score"] = ALPHA * norm_audio + (1 - ALPHA) * norm_text

    best = results.sort_values("combined_score", ascending=False).iloc[0]

    rec_title  = best["track_name"]
    rec_artist = best["track_artist"]

    # ── LLM explanation ───────────────────────────────────────────────────────
    explanation = "(LLM not loaded)"
    if _llm is not None:
        input_profile = _build_profile(song, artist, f"Song: {song}\nArtist: {artist}")
        rec_profile   = _build_profile(rec_title, rec_artist, f"Song: {rec_title}\nArtist: {rec_artist}")

        prompt = PROMPT_TEMPLATE.format(
            input_profile       = input_profile,
            recommended_profile = rec_profile,
            audio_sim           = round(float(best["audio_similarity"]), 4),
            text_sim            = round(float(best["text_similarity"]), 4),
            combined            = round(float(best["combined_score"]), 4),
        )
        explanation = _llm.invoke(prompt).content.strip()

    return {
        "recommended_song":   rec_title,
        "recommended_artist": rec_artist,
        "audio_similarity":   round(float(best["audio_similarity"]), 4),
        "text_similarity":    round(float(best["text_similarity"]), 4),
        "combined_score":     round(float(best["combined_score"]), 4),
        "explanation":        explanation,
    }


# ── Retrieval recommendation ────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a music recommendation assistant. Explain why the recommended song \
is a good match for the input song in a natural, conversational way.

You have been given the full profiles of both songs including their lyrics, \
genre, and audio characteristics. Use this data to inform your explanation, \
but DO NOT mention any numbers, scores, or decimal values in your response. \
Instead, interpret them in plain language:
  - high energy (> 0.7) → "energetic", "lively", "intense"
  - low energy (< 0.4)  → "calm", "mellow", "laid-back"
  - high valence (> 0.7) → "upbeat", "joyful", "feel-good"
  - low valence (< 0.4)  → "melancholic", "dark", "emotional"
  - high danceability (> 0.7) → "great for dancing", "groovy", "rhythmic"
  - high acousticness (> 0.7) → "acoustic", "organic", "stripped-back"
  - high instrumentalness (> 0.5) → "instrumental", "minimal vocals"
  - high speechiness (> 0.6) → "rap", "spoken word", "lyric-heavy"

━━━ INPUT SONG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{input_profile}

━━━ RECOMMENDED SONG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{recommended_profile}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write a short, friendly explanation (3–5 sentences) of why the recommended \
song is a great match. Focus on shared mood, energy, genre, lyrical themes, \
and style. Never mention numbers, scores, or decimal values — speak as if you \
are recommending a song to a friend.\
"""


def _build_profile(title: str, artist: str, doc_content: str) -> str:
    """Add readable audio features to a song's document content."""
    lines = [doc_content.strip()]
    mask  = _df["_title_lower"] == title.lower().strip()
    if mask.sum() > 0:
        row       = _df[mask].iloc[0]
        audio_str = "  |  ".join(
            f"{col}: {row[col]:.3f}" for col in AUDIO_COLS if col in _df.columns
        )
        lines.append(f"Audio features: {audio_str}")
    return "\n".join(lines)


def _retrieval_recommend(song: str, artist: str,
                   name: str, chroma, embeddings) -> dict:
    query   = f"{song} by {artist}"
    try:
        results = chroma.similarity_search_with_relevance_scores(
            query,
            k=CANDIDATE_K,
            filter={"title": {"$ne": song.lower()}},
        )
        print(f"[DEBUG] {name}: got {len(results)} results for '{query}'")
    except Exception as e:
        print(f"[DEBUG] {name}: search failed with error: {e}")
        return {"error": str(e)}

    if not results:
        print(f"[DEBUG] {name}: results list is empty")
        return {"error": "No candidates retrieved."}

    input_audio = _get_audio_vec(song, artist)

    fused = []
    for doc, text_sim in results:
        cand_title  = doc.metadata.get("title", "")
        cand_artist = doc.metadata.get("artist", "")
        cand_audio  = _get_audio_vec(cand_title, cand_artist)

        if input_audio is not None and cand_audio is not None:
            audio_sim = float(max(0.0, cosine_similarity(input_audio, cand_audio)[0][0]))
        else:
            audio_sim = 0.0

        fused.append((doc, text_sim, audio_sim))

    # Normalise each signal to [0, 1] across the candidate pool so that
    # ALPHA truly controls the balance rather than raw score magnitudes.
    audio_scores = [x[2] for x in fused]
    text_scores  = [x[1] for x in fused]
    audio_min, audio_max = min(audio_scores), max(audio_scores)
    text_min,  text_max  = min(text_scores),  max(text_scores)

    fused = [
        (doc, text_sim, audio_sim,
        ALPHA * (audio_sim - audio_min) / (audio_max - audio_min + 1e-9)
        + (1 - ALPHA) * (text_sim - text_min) / (text_max - text_min + 1e-9))
        for doc, text_sim, audio_sim in fused
    ]

    fused.sort(key=lambda x: x[3], reverse=True)
    best_doc, best_text, best_audio, best_combined = fused[0]

    rec_title  = best_doc.metadata.get("title", "?").title()
    rec_artist = best_doc.metadata.get("artist", "?").title()

    input_docs = chroma.similarity_search(
        query, k=1, filter={"title": song.lower()}
    )
    input_content = input_docs[0].page_content if input_docs else f"Song: {song}\nArtist: {artist}"

    prompt = PROMPT_TEMPLATE.format(
        input_profile       = _build_profile(song, artist, input_content),
        recommended_profile = _build_profile(rec_title, rec_artist, best_doc.page_content),
        audio_sim           = best_audio,
        text_sim            = best_text,
        combined            = best_combined,
    )

    explanation = _llm.invoke(prompt).content.strip() if _llm is not None else "(LLM not loaded — run without --no-llm for explanations)"

    return {
        "recommended_song":   rec_title,
        "recommended_artist": rec_artist,
        "audio_similarity":   round(best_audio, 4),
        "text_similarity":    round(best_text, 4),
        "combined_score":     round(best_combined, 4),
        "explanation":        explanation,
    }


# ── Public API ────────────────────────────────────────────────────────────────
def recommend(song: str, artist: str) -> dict:
    """
    Return recommendations from all three methods for a given song.

    Parameters
    ----------
    song   : song title (must exist in the dataset)
    artist : artist name

    Returns
    -------
    dict with keys "knn", "retrieval_minilm", "retrieval_mpnet"
    """
    _check_loaded()

    return {
        "knn":        _knn_recommend(song, artist),
        "retrieval_minilm": _retrieval_recommend(song, artist, "MiniLM", _chroma_a, _embeddings_a),
        "retrieval_mpnet":  _retrieval_recommend(song, artist, "mpnet",  _chroma_b, _embeddings_b),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Music recommender — query all three methods")
    parser.add_argument("song",     help="Song title (must exist in the dataset)")
    parser.add_argument("artist",   help="Artist name")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM — retrieval and scores only")
    args = parser.parse_args()

    load_models(use_llm=not args.no_llm)
    results = recommend(args.song, args.artist)

    for method, result in results.items():
        print(f"\n{'='*55}")
        print(f"  {method.upper()}")
        print(f"{'='*55}")
        print(f"  Recommended : {result['recommended_song']} — {result['recommended_artist']}")
        print(f"  Audio sim   : {result['audio_similarity']}")
        print(f"  Text sim    : {result['text_similarity']}")
        print(f"  Combined    : {result['combined_score']}")
        if "explanation" in result:
            print(f"\n  {result['explanation']}")
