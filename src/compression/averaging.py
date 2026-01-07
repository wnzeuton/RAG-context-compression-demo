import numpy as np
from src.compression.base import Compressor

class AveragingCompressor(Compressor):
    """
    Compress chunks by averaging embeddings in groups determined by 'level'.
    """

    def compress(self, chunks, level=1):
        """
        Args:
            chunks (List[dict]): Must have 'embedding' key
            level (int): Number of groups to compress into (1 = fully average all chunks)

        Returns:
            List[dict]: Compressed chunks
        """
        if len(chunks) == 0:
            return []

        # Ensure level is valid
        n_chunks = len(chunks)
        n_groups = max(1, min(level, n_chunks))

        # Split chunks into roughly equal groups
        group_size = n_chunks // n_groups
        compressed = []

        for i in range(n_groups):
            start = i * group_size
            end = (i + 1) * group_size if i < n_groups - 1 else n_chunks
            group = chunks[start:end]

            # Average embeddings
            embeddings = np.array([c["embedding"] for c in group])
            avg_emb = embeddings.mean(axis=0)

            # Merge text (optional: join texts for demo purposes)
            combined_text = " ".join([c["text"] for c in group])

            compressed.append({
                "doc_id": f"group_{i}",
                "text": combined_text,
                "embedding": avg_emb
            })

        return compressed