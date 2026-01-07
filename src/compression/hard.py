import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
from src.compression.base import Compressor
from src.compression.summarizer import summarize_local_safe

# Initialize once at module level to avoid re-init crashes
# Force CPU to rule out Metal/MPS driver conflicts on Mac
_MODEL = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

class HardCompressor(Compressor):
    def __init__(self):
        # Use the global model
        self.model = _MODEL

    def compress(self, chunks, ratio=1.0):
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

            # Combine group text
            combined_text = " ".join([c["text"] for c in group])

            summary_text = summarize_local_safe(combined_text)
            # summary_text = "Nights by Frank Ocean"
            
            # Embed the summary
            summary_emb = self.model.encode(summary_text)
            # summary_emb = []

            compressed.append({
                "doc_id": f"summary_{i}",
                "text": summary_text,
                "embedding": summary_emb
            })

        return compressed