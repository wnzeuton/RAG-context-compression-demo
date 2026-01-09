from pathlib import Path
import numpy as np

from src.ingestion.chunker import chunk_documents_from_files
from src.ingestion.embedder import embed_chunks
from src.retrieval.faiss_index import build_faiss_index, save_index


# =====================================================
# Paths
# =====================================================
RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")

file_paths = list(RAW_DIR.glob("*.txt"))


# =====================================================
# Chunk documents
# =====================================================
chunks = chunk_documents_from_files(file_paths)
print(f"Created {len(chunks)} chunks from {len(file_paths)} files")


# =====================================================
# Embed chunks
# =====================================================
embeddings, chunks_with_meta = embed_chunks(chunks)
print(f"Created embeddings of shape {embeddings.shape}")

# Ensure float32 for FAISS
embeddings = embeddings.astype("float32")


# =====================================================
# Build FAISS index
# =====================================================
index = build_faiss_index(embeddings)
print("FAISS index built")


# =====================================================
# Save index + metadata
# =====================================================
save_index(
    index=index,
    chunks=chunks_with_meta,
    output_dir=INDEX_DIR
)

print("FAISS index and chunk metadata saved")