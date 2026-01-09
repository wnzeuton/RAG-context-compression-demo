import numpy as np
from sentence_transformers import SentenceTransformer


def format_chunk_for_embedding(chunk: dict) -> str:
    """
    Inject metadata directly into the embedded text.
    """
    parts = []

    if chunk.get("title"):
        parts.append(f"Title: {chunk['title']}")

    if chunk.get("type"):
        parts.append(f"Type: {chunk['type']}")

    # Optional but recommended if you want section awareness
    if "section" in chunk:
        parts.append(f"Section: {chunk['section']}")

    parts.append("")  # blank line before body
    parts.append(chunk["text"])

    return "\n".join(parts)


def embed_chunks(
    chunks,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str | None = None,
):
    """
    Embed chunks with metadata-aware text.

    Returns:
        embeddings: np.ndarray
        chunks_with_meta: list[dict]
    """
    model = SentenceTransformer(model_name, device=device)

    texts = []
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        text = format_chunk_for_embedding(chunk)
        texts.append(text)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    enriched_chunks = []
    for chunk, emb, text in zip(chunks, embeddings, texts):
        enriched_chunks.append({
            **chunk,
            "embedded_text": text,
            "embedding": emb.tolist()  # convert to list for JSON serialization
        })

    return embeddings, enriched_chunks