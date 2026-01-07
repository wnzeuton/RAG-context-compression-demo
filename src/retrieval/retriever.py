import numpy as np
from src.retrieval.faiss_index import load_faiss_index, load_chunk_metadata
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL, TOP_K

class Retriever:
    def __init__(self, top_k=TOP_K):
        self.index = load_faiss_index()
        self.chunks = load_chunk_metadata()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.top_k = top_k

    def query(self, text):
        query_emb = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_emb).astype("float32"), self.top_k)
        results = [self.chunks[i] for i in indices[0]]

        return results