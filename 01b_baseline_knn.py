"""
STEP 1b — Baseline Music Recommender (k-NN with Cosine Similarity)

This is the baseline recommender model. It uses a k-Nearest Neighbours (k-NN)
algorithm to find the most similar songs to a given target, computing distance
with Cosine Similarity on numeric audio features.

Compared to the original project:
  - Euclidean and Manhattan distance metrics have been removed.
  - Only Cosine Similarity is used, as it is the most appropriate metric for
    high-dimensional feature vectors (it is scale-invariant and focuses on
    the angle between vectors rather than their magnitude).

Dataset
-------
The original project used the Million Song Dataset (MSD) summary file.
That dataset has been replaced with a custom CSV file that needs to be provided.

Expected CSV format (place the file at data/songs_dataset.csv):

  title,artist,<feature_1>,<feature_2>,...,<feature_n>

  - title  : song title (string)
  - artist : artist name (string)
  - All remaining columns must be numeric audio features
    (e.g. tempo, energy, danceability, valence, loudness, etc.)
  - There is no restriction on the number of feature columns.
  - A lyrics feature will also be included

TODO: Add your dataset at  data/songs_dataset.csv  before running this script.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/songs_dataset.csv"   # ← Place your CSV here
TOP_K = 10                                  # Number of recommendations to return (1 for now for fastest evaluation from users)

# ── Load dataset ──────────────────────────────────────────────────────────────
# TODO: Replace with your own dataset.
#       The file should be a CSV with columns: title, artist, <numeric features...>
#       Example columns: title, artist, tempo, energy, danceability, valence, loudness

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at '{DATASET_PATH}'.\n"
        "Please add your CSV file before running this script.\n"
        "Expected format: title,artist,<feature_1>,<feature_2>,...,<feature_n>"
    )

df = pd.read_csv(DATASET_PATH)

# Drop duplicates to avoid repeated recommended songs
df = df.drop_duplicates(subset=["track_name", "track_artist"])

# Separate metadata from numeric features
META_COLS = ["track_name", "track_artist"]
# Audio feature columns
feature_cols = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

print(f"Loaded {len(df)} songs with features: {feature_cols}")

# ── Normalize features ────────────────────────────────────────────────────────
scaler = StandardScaler()
features = scaler.fit_transform(df[feature_cols].fillna(0))

# ── k-NN Recommender (Cosine Similarity only) ─────────────────────────────────
def recommend(target_title, target_artist=None, k=TOP_K):
    """
    Find the k most similar songs to the target using Cosine Similarity.

    Parameters
    ----------
    target_title  : exact song title (case-insensitive)
    target_artist : optional artist filter to resolve duplicates
    k             : number of recommendations to return
    mask = df["track_name"].str.lower() == target_title.lower()

    Returns
    -------
    DataFrame with columns [title, artist, cosine_similarity]
    """
    mask = df["track_name"].str.lower() == target_title.lower()

    if target_artist:
        mask &= (
            df["track_artist"].str.lower()
            == target_artist.lower()
        )

    if mask.sum() == 0:
        raise ValueError(f"Song '{target_title}' not found.")

    target_idx = df[mask].index[0]

    target_vec = features[target_idx].reshape(1, -1)

    similarities = cosine_similarity(target_vec, features)[0]

    results = df[META_COLS].copy()
    results["cosine_similarity"] = similarities

    # Remove the target song itself
    results = results.drop(index=target_idx)

    # Sort by similarity
    results = results.sort_values("cosine_similarity", ascending=False).head(k)

    return results.reset_index(drop=True)

# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    TARGET_SONG = "Shape of You"
    TARGET_ARTIST = "Ed Sheeran"

    recs = recommend(TARGET_SONG, TARGET_ARTIST)

    print("\nRecommendations:\n")
    print(recs.to_string(index=False))