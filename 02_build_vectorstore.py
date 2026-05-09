"""
STEP 2 — Build ChromaDB Vector Stores (Two Embedding Models)

This script reads the collected song lyrics (JSON files from Step 1) and
builds TWO separate ChromaDB vector stores — one per embedding model — so
that we can later compare which embedding strategy produces better
music recommendations.

Architecture overview
─────────────────────
  Model A  →  Embedding type A  →  ChromaDB at data/chroma_db_model_a/
  Model B  →  Embedding type B  →  ChromaDB at data/chroma_db_model_b/

The two vector stores feed into Step 3, where each is used inside a full
RAG pipeline and the results are compared against the k-NN baseline.

TODO — Choose and configure your two embedding models
──────────────────────────────────────────────────────
  Replace the placeholder sections marked with  # ← TODO  below.
  Two common options to mix and match:

  Option 1 · Local HuggingFace model (no API key needed)
      from langchain_community.embeddings import HuggingFaceEmbeddings
      embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

  Option 2 · OpenAI embeddings
      from langchain_openai import OpenAIEmbeddings
      embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

  Option 3 · Cohere embeddings
      from langchain_cohere import CohereEmbeddings
      embeddings = CohereEmbeddings(model="embed-english-v3.0")

  You can also experiment with domain-specific or multilingual models.
  Document your choice and rationale in README.md.
"""
import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/songs_dataset.csv"

CHROMA_DIR_A = "data/chroma_db_minilm"
CHROMA_DIR_B = "data/chroma_db_mpnet"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── HuggingFace Embedding Models ─────────────────────────────────────────────
# Model A — Fast and lightweight
embeddings_a = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_A_NAME = "all-MiniLM-L6-v2"

# Model B — Larger and more accurate
embeddings_b = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

EMBEDDING_B_NAME = "all-mpnet-base-v2"

# ── Load Dataset ──────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at '{DATASET_PATH}'."
    )

df = pd.read_csv(DATASET_PATH)

# Remove duplicate songs
df = df.drop_duplicates(
    subset=["track_name", "track_artist"]
)

# Fill missing text fields
text_cols = [
    "track_name",
    "track_artist",
    "playlist_genre",
    "playlist_subgenre",
    "track_album_name",
]

for col in text_cols:
    df[col] = df[col].fillna("")

# ── Create Documents ──────────────────────────────────────────────────────────
documents = []

for _, row in df.iterrows():

    # Create semantic text for embeddings
    content = f"""
    Song: {row['track_name']}
    Artist: {row['track_artist']}
    Album: {row['track_album_name']}
    Genre: {row['playlist_genre']}
    Subgenre: {row['playlist_subgenre']}
    """

    doc = Document(
        page_content=content,
        metadata={
            "title": str(row["track_name"]).lower(),
            "artist": str(row["track_artist"]).lower(),
        },
    )

    documents.append(doc)

print(f"Loaded {len(documents)} songs.")

# ── Split Documents ───────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")

# ── Helper Function ───────────────────────────────────────────────────────────
def build_vectorstore(embeddings, chroma_dir: str, name: str):

    print(f"\nBuilding vector store for {name}...")

    os.makedirs(chroma_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )

    vectorstore.persist()

    print(f"✓ Saved vector store to '{chroma_dir}'")

# ── Build Both Vector Stores ──────────────────────────────────────────────────
build_vectorstore(
    embeddings_a,
    CHROMA_DIR_A,
    EMBEDDING_A_NAME,
)

build_vectorstore(
    embeddings_b,
    CHROMA_DIR_B,
    EMBEDDING_B_NAME,
)

print("\nDone! Both HuggingFace vector stores were created successfully.")