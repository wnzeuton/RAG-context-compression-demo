import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
from src.compression.base import Compressor
from src.compression.summarizer import summarize_local_safe
from collections import defaultdict

# Initialize once at module level to avoid re-init crashes
# Force CPU to rule out Metal/MPS driver conflicts on Mac
_MODEL = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

class HardCompressor(Compressor):
    def __init__(self):
        # Use the global model
        self.model = _MODEL

    def compress(self, chunks, level=1.0):
        if len(chunks) == 0:
            return []

        # Group chunks by document title
        chunks_by_doc = defaultdict(list)
        for c in chunks:
            doc_title = c.get("title", "unknown")
            chunks_by_doc[doc_title].append(c)

        compressed = []
        
        # Calculate max_length based on level (inverse relationship)
        # level 1.0 = max 200 tokens (least compression), level 0.1 = max 50 tokens (most compression)
        max_length = int(50 + level * 150)

        # Process each document separately
        for doc_title, doc_chunks in chunks_by_doc.items():
            # Combine all chunks from this document
            combined_text = " ".join([c["text"] for c in doc_chunks])
            
            # Summarize the combined text
            summary_text = summarize_local_safe(combined_text, max_length=max_length)
            
            # Embed the summary
            summary_emb = self.model.encode(summary_text)

            compressed.append({
                "doc_id": f"summary_{doc_title}",
                "title": doc_title,
                "text": summary_text,
                "embedding": summary_emb
            })

        return compressed