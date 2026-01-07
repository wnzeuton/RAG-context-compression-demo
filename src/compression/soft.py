import numpy as np
from src.compression.base import Compressor

class SoftCompressor(Compressor):
    """
    Soft compression using mean pooling with a compression ratio.
    """
    def compress(self, chunks, ratio=1.0):
        """
        Args:
            chunks (List[dict]): Each must have 'embedding'
            ratio (float): 0 < ratio <= 1. 1=no compression, 0.1=10% of chunks

        Returns:
            List[dict]: Compressed chunks with averaged embeddings
        """
        if len(chunks) == 0:
            return []

        n_chunks = len(chunks)
        n_output = max(1, round(n_chunks * ratio))
        group_size = n_chunks // n_output

        compressed = []
        for i in range(n_output):
            start = i * group_size
            end = (i + 1) * group_size if i < n_output - 1 else n_chunks
            group = chunks[start:end]

            embeddings = np.array([c["embedding"] for c in group])
            avg_emb = embeddings.mean(axis=0)

            combined_text = " ".join([c["text"] for c in group])
            compressed.append({
                "doc_id": f"group_{i}",
                "text": combined_text,
                "embedding": avg_emb
            })

        return compressed