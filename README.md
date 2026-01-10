# RAG Context Compression Demo

This is a **Streamlit demo** illustrating **Retrieval-Augmented Generation (RAG)** and **hard context compression** in a controlled setting.  
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

- **FAISS** finds the top 10 most relevant chunks for a query  
- **Sentence embeddings** are generated with `sentence-transformers/all-MiniLM-L6-v2` for both document chunks and the query  
- **% relevance** = number of retrieved chunks from a document ÷ total chunks in that document  
- High % relevance can appear even for unrelated queries (retrieval is relative)

---

## Hard Compression

- Summarizes retrieved chunks using the **`sshleifer/distilbart-cnn-12-6`** model  
- Reduces context size → lower latency  
- Higher compression may remove fine-grained details  
- Only **compressed summaries** are sent to the LLM

---

## LLM

- Uses **Qwen3-0.6B**, a causal language model  
- Generation is deterministic (`do_sample=False`) to reduce output variance  
- Receives **only the context determined by retrieval and compression** (except in No RAG mode)

---

## Context Sent to the LLM

- Shows the **exact text** sent after retrieval and optional compression  
- Critical for understanding why an answer succeeds or fails

---

## Running the Demo
**IMPORTANT**: Ensure that the `.txt` files in `data/raw` are downloaded and available locally. Alternatively, you can use your own **similarly formatted** documents for RAG.

```bash
pip install -r requirements.txt
python3 -m scripts.build_index
streamlit run app.py
