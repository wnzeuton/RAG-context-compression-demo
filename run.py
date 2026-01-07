from src.compression.summarizer import summarize_local_safe
import os
import torch


from src.retrieval.retriever import Retriever

retriever = Retriever()
chunks = retriever.query("What is space?")
print(len(chunks))  # check how many chunks you got

# Soft compression test
from src.compression.soft import SoftCompressor
compressor = SoftCompressor()
soft_chunks = compressor.compress(chunks, ratio=0.5)
print(len(soft_chunks))

# Hard compression test
from src.compression.hard import HardCompressor
hard_compressor = HardCompressor()
hard_chunks = hard_compressor.compress(chunks, ratio=0.1)
print(hard_chunks);
