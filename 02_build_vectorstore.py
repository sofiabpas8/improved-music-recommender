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

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/songs_dataset.csv"   # ← same CSV used by the k-NN baseline
CHROMA_DIR_A = "data/chroma_db_model_a"
CHROMA_DIR_B = "data/chroma_db_model_b"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100

# ── Embedding models ──────────────────────────────────────────────────────────
# TODO: Import and instantiate your two chosen embedding models here.
#       Delete the two NotImplementedError lines once you have done so.

# Model A — replace this block ↓
# Example:
#   from langchain_community.embeddings import HuggingFaceEmbeddings
#   embeddings_a = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings_a = None  # ← TODO: replace with your Model A instantiation
EMBEDDING_A_NAME = "MODEL_A"  # ← TODO: set a short descriptive name, e.g. "MiniLM-L6-v2"

# Model B — replace this block ↓
# Example:
#   from langchain_openai import OpenAIEmbeddings
#   embeddings_b = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings_b = None  # ← TODO: replace with your Model B instantiation
EMBEDDING_B_NAME = "MODEL_B"  # ← TODO: set a short descriptive name, e.g. "OpenAI-text-embedding-3-small"

# ── Load lyrics from CSV ──────────────────────────────────────────────────────
# The CSV must have at minimum: title, artist, lyrics
# Any additional numeric columns (audio features) are ignored here —
# they are used only by the k-NN baseline in 01b_baseline_knn.py.

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at '{DATASET_PATH}'.\n"
        "Please add your CSV file before running this script.\n"
        "Required columns: title, artist, lyrics (plus any numeric feature columns)"
    )

df = pd.read_csv(DATASET_PATH)

documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=str(row["lyrics"]),
        metadata={
            "title":  str(row["title"]).lower(),
            "artist": str(row["artist"]).lower(),
        },
    )
    documents.append(doc)

print(f"Loaded {len(documents)} songs.")

# ── Split into chunks ─────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")


# ── Helper: build and persist one vector store ────────────────────────────────
def build_vectorstore(embeddings, chroma_dir: str, name: str):
    if embeddings is None:
        print(f"\n⚠  Skipping {name} — embedding model not configured yet (see TODO above).")
        return

    print(f"\nBuilding vector store for {name} → {chroma_dir} ...")
    os.makedirs(chroma_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vectorstore.persist()
    print(f"✓ {name} vector store saved to '{chroma_dir}/'")


# ── Build both stores ─────────────────────────────────────────────────────────
build_vectorstore(embeddings_a, CHROMA_DIR_A, EMBEDDING_A_NAME)
build_vectorstore(embeddings_b, CHROMA_DIR_B, EMBEDDING_B_NAME)

print("\nDone! Run 03_rag_pipeline.py to evaluate both models.")