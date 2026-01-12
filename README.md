# RAG Context Compression Demo

This is a **[Streamlit Demo](https://huggingface.co/spaces/wnzeuton/RAG-context-compression-demo)** illustrating **Retrieval-Augmented Generation (RAG)** and **hard context compression** in a controlled setting.  
It is designed to help explore how retrieval, compression, and grounding constraints affect LLM outputs.

---

## Overview

- User query → retrieves relevant document chunks → optionally compressed → sent to the LLM (Qwen3-0.6B)  
- The demo shows:
  - Retrieved chunks and document-level relevance
  - Compressed context actually sent to the LLM
  - Final LLM output  
- Goal: observe trade-offs between **latency, detail retention, and grounding fidelity**.

---

## Questions This Demo Explores

- How much detail is lost under aggressive hard compression?
- When does retrieval relevance become misleading?
- How often does a small LLM hallucinate under Loose RAG?
- How does Tight RAG affect answer completeness vs correctness?

---

## Generation Modes

| Mode | Description |
|------|-------------|
| **No RAG (LLM-only)** | LLM answers using only pretrained knowledge. No documents provided. |
| **Loose RAG** | Retrieved context is supporting info; LLM may also rely on general knowledge. |
| **Tight RAG** | LLM must answer **only** using retrieved context; replies *“I don’t know”* if answer is not present. |

---

## Document Corpus

- 5 **synthetic Wikipedia-style articles** about **fictional individuals**
- Ensures the LLM must rely on retrieval; prevents pretraining leakage

---

## Retrieval & Relevance

- **Embeddings** are generated with `sentence-transformers/all-MiniLM-L6-v2` for both document chunks and the query  
- **FAISS** finds the top 10 most relevant chunks for a query using cosine similarity


---

## Hard Compression

- Summarizes retrieved chunks using the `sshleifer/distilbart-cnn-12-6` model  
- Reduces context size → lower latency  
- Higher compression may remove fine-grained details  
- Only **compressed summaries** are sent to the LLM

---

## LLM

- Uses **Qwen3-0.6B**, a causal language model  
- Generation is deterministic to reduce output variance  
- Enforces varying RAG modes via system prompting
  
---

## Limitations

- Compression is abstractive and may introduce summarization bias
- FAISS relevance is relative, not absolute
- Results are specific to small models (Qwen3-0.6B)
- Synthetic documents simplify real-world retrieval noise

## Running the Demo
**IMPORTANT**: Ensure that the `.txt` files in `data/raw` are downloaded and available locally. Alternatively, you can use your own **similarly formatted** documents for RAG.

```bash
pip install -r requirements.txt
python3 -m scripts.build_index
streamlit run app.py
