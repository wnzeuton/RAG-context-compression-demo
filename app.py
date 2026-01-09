import streamlit as st
import torch
import time
from pathlib import Path
from collections import defaultdict

from src.retrieval.retriever import Retriever
from src.compression.soft import SoftCompressor
from src.compression.hard import HardCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# App setup
# =====================================================
st.set_page_config(page_title="Context Compression Demo")
st.title("Context Compression Demo")

# =====================================================
# RAG system prompts
# =====================================================
TIGHT_RAG_PROMPT = (
    "You are an assistant that answers questions ONLY using the provided context. "
    "Do not use prior knowledge. "
    "If the answer is not explicitly stated in the context, say 'I don't know.' "
    "Do not mention or reference the context."
)

LOOSE_RAG_PROMPT = (
    "You are a helpful assistant. Use the provided context as supporting information "
    "when relevant, but you may rely on your general knowledge if needed. "
    "Do not explicitly mention the context unless necessary."
)

# =====================================================
# Load full documents
# =====================================================
RAW_DIR = Path("data/raw")

@st.cache_data
def load_full_documents():
    docs = {}
    for path in RAW_DIR.glob("*.txt"):
        docs[path.name] = path.read_text()
    return docs

full_docs = load_full_documents()

# =====================================================
# Initialize modules
# =====================================================
retriever = Retriever()
soft_compressor = SoftCompressor()
hard_compressor = HardCompressor()

# =====================================================
# Load Qwen3-0.6B
# =====================================================
@st.cache_resource
def load_qwen3():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_qwen3()

# =====================================================
# LLM query function
# =====================================================
def query_qwen3(context, question, system_prompt, rag_mode):
    if rag_mode.startswith("Tight"):
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer using ONLY the context:"
        )
    else:
        user_content = (
            f"Here is some relevant information:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

# =====================================================
# Highlight helper
# =====================================================
def highlight_chunks(doc_text, chunks):
    highlighted = doc_text
    chunks = sorted(chunks, key=lambda c: c["start_char"], reverse=True)
    for c in chunks:
        start, end = c["start_char"], c["end_char"]
        snippet = highlighted[start:end]
        highlighted = (
            highlighted[:start]
            + f"<mark style='background-color:#ffe066'>{snippet}</mark>"
            + highlighted[end:]
        )
    return highlighted

# =====================================================
# UI Controls
# =====================================================
query = st.text_input("Enter your query", key="query_input")

rag_mode = st.selectbox(
    "RAG Mode",
    ["Tight (context-only)", "Loose (context-guided)"],
    key="rag_mode_selectbox"
)

compression_type = st.selectbox(
    "Compression method",
    ["Soft (mean-pooled embeddings)", "Hard (summarize then embed)"],
    key="compression_type_selectbox"
)

ratio = st.slider(
    "Compression ratio",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05,
    key="compression_ratio_slider"
)

# =====================================================
# Advanced settings
# =====================================================
with st.expander("Advanced settings"):
    override_prompt = st.checkbox("Override system prompt", key="override_prompt_checkbox")

    system_prompt = (
        TIGHT_RAG_PROMPT
        if rag_mode.startswith("Tight")
        else LOOSE_RAG_PROMPT
    )

    custom_prompt = st.text_area(
        "System prompt",
        value=system_prompt,
        height=150,
        disabled=not override_prompt,
        key="custom_system_prompt_textarea"
    )

    if override_prompt:
        system_prompt = custom_prompt
        st.warning("⚠️ Custom system prompt active")

st.markdown("---")

# =====================================================
# Generate button
# =====================================================
if st.button("Generate", key="generate_button"):
    if not query:
        st.warning("Please enter a query before generating.")
    else:
        start = time.time()
        st.info("Running retrieval + compression + LLM...")

        # -------------------------
        # Retrieval (fixed top_k = 10)
        # -------------------------
        relevant_chunks = retriever.query(query)[:10]

        # -------------------------
        # Compression
        # -------------------------
        compressed = []
        if relevant_chunks:
            if compression_type.startswith("Soft"):
                compressed = soft_compressor.compress(relevant_chunks, ratio)
            else:
                compressed = hard_compressor.compress(relevant_chunks, ratio)

        # -------------------------
        # Build context
        # -------------------------
        context = "\n".join(c["text"] for c in compressed) if compressed else ""

        # -------------------------
        # LLM generation
        # -------------------------
        answer = query_qwen3(
            context=context,
            question=query,
            system_prompt=system_prompt,
            rag_mode=rag_mode
        )

        elapsed = round(time.time() - start, 2)
        st.success(f"Done in {elapsed}s")

        # =====================================================
        # Output
        # =====================================================
        st.subheader("LLM Output")
        st.write(answer)

        # =====================================================
        # Document-level attribution view (updated)
        # =====================================================
        if full_docs:
            st.subheader("Retrieved Evidence in Full Documents")

            # Build relevant chunk mapping
            relevant_chunks_by_doc = defaultdict(list)
            for c in relevant_chunks:
                relevant_chunks_by_doc[c["doc_name"]].append(c)

            # Prepare documents sorted by number of relevant chunks
            docs_with_relevance = []
            for doc_name, doc_text in full_docs.items():
                num_relevant = len(relevant_chunks_by_doc.get(doc_name, []))
                docs_with_relevance.append((doc_name, doc_text, num_relevant))

            # Sort descending by num_relevant
            docs_with_relevance.sort(key=lambda x: x[2], reverse=True)

            for doc_idx, (doc_name, doc_text, num_relevant) in enumerate(docs_with_relevance):
                # Only show documents with at least one relevant chunk first
                with st.expander(f"📄 {doc_name} — {num_relevant}/10 relevant chunks"):
                    all_chunks = retriever.get_all_chunks_for_doc(doc_name)
                    relevant_starts = set(c["start_char"] for c in relevant_chunks_by_doc.get(doc_name, []))

                    for chunk_idx, c in enumerate(all_chunks):
                        is_relevant = c["start_char"] in relevant_starts
                        label = f"🟢 {c.get('section', f'Chunk {chunk_idx+1}')}" if is_relevant else f"⚪ {c.get('section', f'Chunk {chunk_idx+1}')}"
                        
                        with st.expander(label):
                            st.markdown(
                                    f"<pre style='font-size:14px'>{c['text']}</pre>",
                                    unsafe_allow_html=True
                                )
                                

        # =====================================================
        # Show compressed context
        # =====================================================
        if compressed:
            with st.expander("Compressed Context Sent to LLM"):
                for c_idx, c in enumerate(compressed):
                    st.markdown(
                        f"<pre style='font-size:14px'>{c['text']}</pre>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<hr/>", unsafe_allow_html=True)