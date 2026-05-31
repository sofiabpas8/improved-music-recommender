"""
STEP 2 — Build ChromaDB Vector Stores (Two Embedding Models)

Builds TWO ChromaDB vector stores — one per embedding model — so that
Step 3 can compare which embedding strategy produces better recommendations.

Each document contains song metadata + lyrics. Audio features are kept
numeric and handled separately in Step 3 (late fusion).

Architecture
────────────
  Model A (all-MiniLM-L6-v2)  →  ChromaDB at data/chroma_db_minilm/
  Model B (all-mpnet-base-v2) →  ChromaDB at data/chroma_db_mpnet/
"""

import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/songs_dataset.csv"

CHROMA_DIR_A = "data/chroma_db_minilm"
CHROMA_DIR_B = "data/chroma_db_mpnet"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 80

# ── Embedding models ──────────────────────────────────────────────────────────
# Model A — fast and lightweight
embeddings_a     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_A_NAME = "all-MiniLM-L6-v2"

# Model B — larger and more accurate
embeddings_b     = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
EMBEDDING_B_NAME = "all-mpnet-base-v2"

# ── Load dataset ──────────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"\nDataset not found at '{DATASET_PATH}'.")

df = pd.read_csv(DATASET_PATH)
df = df.drop_duplicates(subset=["track_name", "track_artist"])

TEXT_COLS = ["track_name", "track_artist", "playlist_genre",
             "playlist_subgenre", "track_album_name", "lyrics"]
for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("")

# ── Build documents ───────────────────────────────────────────────────────────
documents = []

for _, row in df.iterrows():
    content = (
        f"Album: {row['track_album_name']}\n"
        f"Genre: {row['playlist_genre']} / {row['playlist_subgenre']}\n"
        f"Lyrics:\n{row['lyrics']}"
    )

    doc = Document(
        page_content=content,
        metadata={
            "title":  str(row["track_name"]).lower(),
            "artist": str(row["track_artist"]).lower(),
        },
    )
    documents.append(doc)

print(f"Loaded {len(documents)} songs.")

# ── Split documents ───────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# ── Build vector store ────────────────────────────────────────────────────────
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

build_vectorstore(embeddings_a, CHROMA_DIR_A, EMBEDDING_A_NAME)
build_vectorstore(embeddings_b, CHROMA_DIR_B, EMBEDDING_B_NAME)

print("\nDone! Both vector stores created successfully.")
