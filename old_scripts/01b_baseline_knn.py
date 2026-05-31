"""
STEP 1b — Baseline Music Recommender (k-NN with Late Fusion)

Recommends songs using cosine similarity on two independent feature spaces,
combined via late fusion:

  combined = ALPHA * audio_sim + (1 - ALPHA) * text_sim

Audio features (numeric)
  Standard-scaled audio attributes: tempo, energy, danceability, valence, etc.
  Cosine similarity is computed directly on the normalised feature vectors.

Text features (TF-IDF)
  Song metadata and lyrics are concatenated into a single document per song,
  then vectorised with TF-IDF (with English stopword removal).
  Cosine similarity is computed on the resulting sparse vectors.

Compared to the original project:
  - Euclidean and Manhattan distance metrics have been removed.
  - Lyrics and metadata are now included via TF-IDF (not only audio features).
  - Late fusion replaces single-space cosine similarity.
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/songs_dataset.csv"
TOP_K        = 10
ALPHA        = 0.5   # weight of audio similarity (1 - ALPHA = weight of text)

# ── Load dataset ──────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at '{DATASET_PATH}'.\n"
        "Expected format: title, artist, <audio features...>, lyrics"
    )

df = pd.read_csv(DATASET_PATH)
df = df.drop_duplicates(subset=["track_name", "track_artist"])
df = df.reset_index(drop=True)

META_COLS = ["track_name", "track_artist"]

# ── Audio features ────────────────────────────────────────────────────────────
AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]

scaler        = StandardScaler()
audio_matrix  = scaler.fit_transform(df[AUDIO_COLS].fillna(0))

print(f"Loaded {len(df)} songs.")

# ── Text features (TF-IDF) ────────────────────────────────────────────────────
TEXT_COLS = ["track_album_name", "playlist_genre", "playlist_subgenre", "lyrics"]

for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("")

def build_text_doc(row) -> str:
    parts = [
        row.get("track_name", ""),
        row.get("track_artist", ""),
        row.get("track_album_name", ""),
        row.get("playlist_genre", ""),
        row.get("playlist_subgenre", ""),
        row.get("lyrics", ""),
    ]
    return " ".join(str(p) for p in parts if p and str(p) != "nan")

text_docs = df.apply(build_text_doc, axis=1).tolist()

tfidf      = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(text_docs)   # sparse matrix

print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

# ── k-NN Recommender (Late Fusion) ────────────────────────────────────────────
def recommend(target_title: str, target_artist: str = None, k: int = TOP_K):
    """
    Find the k most similar songs using late fusion of audio and text similarity.

    Parameters
    ----------
    target_title  : exact song title (case-insensitive)
    target_artist : optional artist filter to resolve duplicates
    k             : number of recommendations to return

    Returns
    -------
    DataFrame with columns [track_name, track_artist,
                             audio_similarity, text_similarity, combined_score]
    """
    mask = df["track_name"].str.lower() == target_title.lower()

    if target_artist:
        mask &= df["track_artist"].str.lower() == target_artist.lower()

    if mask.sum() == 0:
        raise ValueError(f"Song '{target_title}' not found in the dataset.")

    target_idx = df[mask].index[0]

    # Audio similarity
    target_audio = audio_matrix[target_idx].reshape(1, -1)
    audio_sims   = cosine_similarity(target_audio, audio_matrix)[0]

    # Text similarity (linear_kernel is faster than cosine_similarity on sparse)
    target_tfidf = tfidf_matrix[target_idx]
    text_sims    = linear_kernel(target_tfidf, tfidf_matrix)[0]

    results = df[META_COLS].copy()
    results["audio_similarity"] = audio_sims
    results["text_similarity"]  = text_sims
    results = results.drop(index=target_idx)

    # Normalise each signal to [0, 1] across the candidate pool so that
    # ALPHA truly controls the balance rather than raw score magnitudes.
    a_min, a_max = results["audio_similarity"].min(), results["audio_similarity"].max()
    t_min, t_max = results["text_similarity"].min(),  results["text_similarity"].max()
    norm_audio = (results["audio_similarity"] - a_min) / (a_max - a_min + 1e-9)
    norm_text  = (results["text_similarity"]  - t_min) / (t_max  - t_min  + 1e-9)
    results["combined_score"] = ALPHA * norm_audio + (1 - ALPHA) * norm_text
    results = results.sort_values("combined_score", ascending=False).head(k)

    return results.reset_index(drop=True)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    TARGET_SONG   = "Shape of You"
    TARGET_ARTIST = "Ed Sheeran"

    recs = recommend(TARGET_SONG, TARGET_ARTIST)

    print(f"\nTop {TOP_K} recommendations for '{TARGET_SONG}' by {TARGET_ARTIST}:\n")
    print(recs.to_string(index=False))
