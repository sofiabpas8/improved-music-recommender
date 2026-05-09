"""
build.py — Pre-compute and persist all indexes.

Run once before starting the recommender:
    python build.py

Artifacts written to data/:
    songs_clean.csv          — deduplicated dataset used by all methods
    audio_matrix.npy         — StandardScaler-normalised audio feature matrix
    audio_scaler.joblib      — fitted StandardScaler
    tfidf_matrix.npz         — TF-IDF sparse matrix (k-NN text features)
    tfidf_vectorizer.joblib  — fitted TfidfVectorizer
    chroma_db_minilm/        — ChromaDB vector store (MiniLM embeddings)
    chroma_db_mpnet/         — ChromaDB vector store (mpnet embeddings)
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "data/songs_dataset.csv"
CLEAN_PATH    = "data/songs_clean.csv"
CHROMA_DIR_A  = "data/chroma_db_minilm"
CHROMA_DIR_B  = "data/chroma_db_mpnet"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

AUDIO_COLS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo",
]
TEXT_COLS = [
    "track_name", "track_artist", "track_album_name",
    "playlist_genre", "playlist_subgenre", "lyrics",
]

os.makedirs("data", exist_ok=True)

# ── Load & clean dataset ──────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df.drop_duplicates(subset=["track_name", "track_artist"])
df = df.reset_index(drop=True)

for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("")

df.to_csv(CLEAN_PATH, index=False)
print(f"  {len(df)} songs saved to {CLEAN_PATH}")

# ── Audio features ────────────────────────────────────────────────────────────
print("\nBuilding audio index...")
scaler       = StandardScaler()
audio_matrix = scaler.fit_transform(df[AUDIO_COLS].fillna(0))

np.save("data/audio_matrix.npy", audio_matrix)
joblib.dump(scaler, "data/audio_scaler.joblib")
print("  Saved audio_matrix.npy + audio_scaler.joblib")

# ── TF-IDF (k-NN text features) ───────────────────────────────────────────────
print("\nBuilding TF-IDF index...")

def build_text_doc(row) -> str:
    return " ".join(str(row.get(c, "")) for c in TEXT_COLS if row.get(c, ""))

text_docs    = df.apply(build_text_doc, axis=1).tolist()
tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(text_docs)

scipy.sparse.save_npz("data/tfidf_matrix.npz", tfidf_matrix)
joblib.dump(tfidf, "data/tfidf_vectorizer.joblib")
print(f"  Vocabulary: {len(tfidf.vocabulary_)} terms")
print("  Saved tfidf_matrix.npz + tfidf_vectorizer.joblib")

# ── ChromaDB vector stores (RAG A + B) ────────────────────────────────────────
documents = []
for _, row in df.iterrows():
    content = (
        f"Song: {row['track_name']}\n"
        f"Artist: {row['track_artist']}\n"
        f"Album: {row['track_album_name']}\n"
        f"Genre: {row['playlist_genre']} / {row['playlist_subgenre']}\n"
        f"Lyrics:\n{row['lyrics']}"
    )
    documents.append(Document(
        page_content=content,
        metadata={
            "title":  str(row["track_name"]).lower(),
            "artist": str(row["track_artist"]).lower(),
        },
    ))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"\nBuilding ChromaDB stores ({len(chunks)} chunks from {len(documents)} songs)...")

def build_vectorstore(embeddings, chroma_dir: str, name: str):
    print(f"\n  [{name}]")
    os.makedirs(chroma_dir, exist_ok=True)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vs.persist()
    print(f"  Saved to {chroma_dir}")

print("\n  Loading embedding models (first run downloads weights)...")
embeddings_a = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_b = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

build_vectorstore(embeddings_a, CHROMA_DIR_A, "all-MiniLM-L6-v2")
build_vectorstore(embeddings_b, CHROMA_DIR_B, "all-mpnet-base-v2")

print("\n✓ All indexes built. Ready to run recommender.py")
