from pathlib import Path
import json
import faiss
import numpy as np


# =====================================================
# FAISS index utilities
# =====================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using cosine similarity.

    Assumes embeddings are L2-normalized.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product == cosine when normalized
    index.add(embeddings)

    return index


# =====================================================
# Save / load
# =====================================================

def save_index(
    index: faiss.Index,
    chunks: list[dict],
    output_dir: str | Path,
):
    """
    Save FAISS index and chunk metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # Save chunk metadata (parallel array)
    with open(output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_index(
    input_dir: str | Path,
):
    """
    Load FAISS index and chunk metadata.
    """
    input_dir = Path(input_dir)

    index = faiss.read_index(str(input_dir / "index.faiss"))

    with open(input_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


# =====================================================
# Query
# =====================================================

def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = 10,
):
    """
    Perform semantic search.

    Returns:
        list of chunks with similarity scores
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[None, :]

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append({
            **chunks[idx],
            "score": float(score)
        })

    return results