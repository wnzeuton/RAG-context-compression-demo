import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.retrieval.faiss_index import load_index
from src.config import EMBEDDING_MODEL, TOP_K

device = "cuda" if torch.cuda.is_available() else "cpu"

class Retriever:
    def __init__(self, faiss_dir: str | Path = "data/index", top_k: int = TOP_K):
        self.top_k = top_k

        # Load FAISS index + chunk metadata
        self.index, self.chunks = load_index(faiss_dir)

        # Cache embeddings for optional filtering (FAISS already has them)
        self.embeddings = np.array([c["embedding"] for c in self.chunks], dtype="float32")

        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        self.model.to(device)

        # Group all chunks by document for easy access
        self.chunks_by_doc = {}
        for c in self.chunks:
            self.chunks_by_doc.setdefault(c["doc_name"], []).append(c)

    def _build_temp_index(self, embeddings: np.ndarray):
        """Build temporary FAISS index for a subset of chunks (if filtering)"""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity
        index.add(embeddings)
        return index

    def query(
        self,
        query_text: str,
        allowed_entities: set[str] | None = None,
        top_k: int | None = None
    ) -> list[dict]:
        """
        Args:
            query_text: text to search
            allowed_entities: optional whitelist of chunk titles
            top_k: number of results to return

        Returns:
            List of chunks sorted by similarity
        """
        top_k = top_k or self.top_k

        # ----------------------------------
        # Step 1: filter eligible chunks
        # ----------------------------------
        eligible_indices = []
        eligible_chunks = []

        for i, c in enumerate(self.chunks):
            if allowed_entities is not None and c["title"] not in allowed_entities:
                continue
            eligible_indices.append(i)
            eligible_chunks.append(c)

        if not eligible_chunks:
            return []

        eligible_embeddings = self.embeddings[eligible_indices]

        # ----------------------------------
        # Step 2: build temporary FAISS index if needed
        # ----------------------------------
        if len(eligible_chunks) < len(self.chunks):
            index = self._build_temp_index(eligible_embeddings)
        else:
            index = self.index

        # ----------------------------------
        # Step 3: embed query & search
        # ----------------------------------
        query_emb = self.model.encode([query_text], normalize_embeddings=True).astype("float32")

        distances, indices = index.search(query_emb, min(top_k, len(eligible_chunks)))

        # ----------------------------------
        # Step 4: return ranked results
        # ----------------------------------
        results = [eligible_chunks[i] for i in indices[0]]
        return results

    def get_all_chunks_for_doc(self, doc_name: str) -> list[dict]:
        """
        Returns all chunks for a given document in order of start_char
        """
        return sorted(self.chunks_by_doc.get(doc_name, []), key=lambda c: c["start_char"])