import streamlit as st
import torch
import time
from pathlib import Path
from collections import defaultdict

from src.retrieval.retriever import Retriever
from src.compression.hard import HardCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================
# App setup
# =====================================================
st.set_page_config(page_title="RAG Context Compression Demo")
st.title("RAG Context Compression Demo")

with st.expander("What is this demo?", expanded=True):
    st.markdown("""
This demo illustrates **Retrieval-Augmented Generation (RAG)** and **context compression** in a controlled setting.

A user query retrieves relevant text chunks from a small document corpus.  
These chunks may then be **compressed via hard compression** before being sent to a language model for answer generation.

The goal is to observe how **compression level affects latency, faithfulness, and loss of detail** in RAG-based systems.
""")

with st.expander("About the document corpus"):
    st.markdown("""
The document corpus consists of **five synthetic, Wikipedia-style articles** describing **five fictional individuals**: Ilya Moreno, Mei-lin Zhao, Rafael Okoye, Anselma Kruger, and Thomas Albrecht..

These articles were intentionally created to:
- Avoid overlap with the model’s pretraining data
- Ensure the LLM **must rely on retrieved context** to answer questions
- Make the effects of RAG and compression more visible and interpretable
""")

# =====================================================
# System prompts
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

NO_RAG_PROMPT = (
    "You are a helpful assistant. Answer the user's question using your general knowledge."
)

# =====================================================
# Helpers
# =====================================================
def get_default_system_prompt(rag_mode: str) -> str:
    if rag_mode == "No RAG (LLM-only)":
        return NO_RAG_PROMPT
    elif rag_mode.startswith("Tight"):
        return TIGHT_RAG_PROMPT
    else:
        return LOOSE_RAG_PROMPT


# =====================================================
# Load full documents
# =====================================================
RAW_DIR = Path("data/raw")

@st.cache_data
def load_full_documents():
    return {p.name: p.read_text() for p in RAW_DIR.glob("*.txt")}

full_docs = load_full_documents()

# =====================================================
# Initialize modules
# =====================================================
retriever = Retriever()
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
# LLM query
# =====================================================
def query_qwen3(context, question, system_prompt, rag_mode):
    if rag_mode == "No RAG (LLM-only)":
        user_content = f"Question:\n{question}\n\nAnswer:"
    elif rag_mode.startswith("Tight"):
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer using the context:"
        )
    else:
        user_content = (
            f"Here is some relevant information:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# =====================================================
# UI Controls
# =====================================================
query = st.text_input("Enter your query")

rag_mode = st.selectbox(
    "Generation Mode",
    [
        "No RAG (LLM-only)",
        "Loose RAG (context-guided)",
        "Tight RAG (context-only)",
    ],
)

st.caption("""
- **No RAG (LLM-only)**: Model uses only pretrained knowledge  
- **Loose RAG**: Context is helpful but not binding  
- **Tight RAG**: Model may answer *only* from retrieved context
""")

use_compression = st.checkbox("Enable compression", value=True)

st.caption(
    "**Hard compression** summarizes retrieved chunks using a separate encoder–decoder model. "
    "Higher compression lowers latency but may remove fine-grained details."
)

compression_level = st.slider(
    "Compression level",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

# =====================================================
# System prompt state handling
# =====================================================
default_prompt = get_default_system_prompt(rag_mode)

if "last_rag_mode" not in st.session_state:
    st.session_state.last_rag_mode = rag_mode

# Reset prompt when mode changes (unless overridden)
if (
    rag_mode != st.session_state.last_rag_mode
    and not st.session_state.get("override_prompt", False)
):
    st.session_state.system_prompt_text = default_prompt

st.session_state.last_rag_mode = rag_mode

# =====================================================
# Advanced settings
# =====================================================
with st.expander("Advanced settings"):
    if rag_mode == "No RAG (LLM-only)":
        st.info("System prompt is fixed in No-RAG mode.")
        system_prompt = default_prompt
    else:
        override = st.checkbox("Override system prompt", key="override_prompt")
        prompt_text = st.text_area(
            "System prompt",
            height=150,
            key="system_prompt_text",
            value=default_prompt,
            disabled=not override,
        )

        system_prompt = prompt_text if override else default_prompt

        if override:
            st.warning("⚠️ Custom system prompt active")

# =====================================================
# Generate
# =====================================================
if st.button("Generate", use_container_width=True):
    if not query:
        st.warning("Please enter a query.")
    else:
        start = time.time()

        if rag_mode == "No RAG (LLM-only)":
            relevant_chunks = []
            context_chunks = []
        else:
            relevant_chunks = retriever.query(query)[:10]
            context_chunks = (
                hard_compressor.compress(relevant_chunks, compression_level)
                if use_compression
                else relevant_chunks
            )

        context = "\n".join(c["text"] for c in context_chunks)

        answer = query_qwen3(
            context=context,
            question=query,
            system_prompt=system_prompt,
            rag_mode=rag_mode,
        )

        st.success(f"Done in {round(time.time() - start, 2)}s")
        st.subheader("LLM Output")
        st.write(answer)

        # =====================================================
        # Retrieved evidence
        # =====================================================
        if rag_mode != "No RAG (LLM-only)" and full_docs:
            st.subheader("Retrieved Evidence in Full Documents")

            # Build relevant chunk mapping
            relevant_chunks_by_doc = defaultdict(list)
            for c in relevant_chunks:
                relevant_chunks_by_doc[c["doc_name"]].append(c)

            # Prepare documents with relevance percentage
            docs_with_relevance = []
            for doc_name, doc_text in full_docs.items():
                all_chunks = retriever.get_all_chunks_for_doc(doc_name)
                total_chunks = len(all_chunks)
                num_relevant = len(relevant_chunks_by_doc.get(doc_name, []))
                percent_relevant = round((num_relevant / total_chunks) * 100) if total_chunks > 0 else 0
                title = all_chunks[0]["title"] if all_chunks else doc_name
                docs_with_relevance.append((doc_name, title, doc_text, num_relevant, total_chunks, percent_relevant))

            # Sort by percentage descending
            docs_with_relevance.sort(key=lambda x: x[5], reverse=True)

            # Display
            for doc_idx, (doc_name, title, doc_text, num_relevant, total_chunks, percent_relevant) in enumerate(docs_with_relevance):
                # Expander label with title + percentage relevant
                label = f"**{title}** — :blue[{percent_relevant}% relevant]"
                
                with st.expander(label, expanded=False):
                    all_chunks = retriever.get_all_chunks_for_doc(doc_name)
                    relevant_starts = set(c["start_char"] for c in relevant_chunks_by_doc.get(doc_name, []))

                    for chunk_idx, c in enumerate(all_chunks):
                        is_relevant = c["start_char"] in relevant_starts
                        color = "green" if is_relevant else "red"
                        label = f":{color}[{c.get('section', f'Chunk {chunk_idx+1}')}]"
                        with st.expander(label):
                            st.markdown(f"<pre style='font-size:14px'>{c['text']}</pre>", unsafe_allow_html=True)

        # =====================================================
        # Show compressed context
        # =====================================================
        if context_chunks:
            with st.expander(":orange[Context Sent to LLM]"):
                for c_idx, c in enumerate(context_chunks):
                    st.markdown(
                        f"<pre style='font-size:14px'>{c['text']}</pre>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<hr/>", unsafe_allow_html=True)