import streamlit as st
import torch
from src.retrieval.retriever import Retriever
from src.compression.soft import SoftCompressor
from src.compression.hard import HardCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# =====================================================
# App setup
# =====================================================
st.set_page_config(page_title="Context Compression Demo")
st.title("Context Compression Demo")

# =====================================================
# Initialize modules
# =====================================================
retriever = Retriever()
soft_compressor = SoftCompressor()
hard_compressor = HardCompressor()

# =====================================================
# Load Qwen3-0.6B model
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
# Query function using chat template
# =====================================================
def query_qwen3(context, question):
    messages = [
        {
            "role": "system",
            "content": (
            "You are an assistant that answers questions ONLY using the provided context. "
            "Do not use prior knowledge. "
            "If there is no context, or if the context does not contain the answer, say 'I don't know.' "
            "Do not mention or reference the context."
            )
        },
        {
            "role": "user",
            "content": (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
            )
        }
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

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return content

# =====================================================
# UI Controls
# =====================================================
query = st.text_input("Enter your query")

compression_type = st.selectbox(
    "Compression method",
    ["Soft (mean-pooled embeddings)", "Hard (summarize then embed)"]
)

ratio = st.slider(
    "Compression ratio",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05
)

top_k = st.slider(
    "Number of retrieved chunks",
    min_value=0,
    max_value=20,
    value=10
)

st.markdown("---")

# =====================================================
# Generate button
# =====================================================
if st.button("Generate"):
    if not query:
        st.warning("Please enter a query before generating.")
    else:
        start = time.time()
        st.info("Running retrieval + compression + LLM...")

        # -------------------------
        # Retrieval
        # -------------------------
        chunks = retriever.query(query)[:top_k]

        # =====================================================
        # Retrieval
        # =====================================================
        chunks = []
        if top_k > 0:  # Only retrieve if user requests chunks
            chunks = retriever.query(query)[:top_k]

        # =====================================================
        # Compression
        # =====================================================
        compressed = []
        if chunks:
            if compression_type.startswith("Soft"):
                compressed = soft_compressor.compress(chunks, ratio)
            else:
                compressed = hard_compressor.compress(chunks, ratio)

        # =====================================================
        # Build context for LLM
        # =====================================================
        context = "\n".join([c["text"] for c in compressed]) if compressed else ""
        answer = query_qwen3(context, query)

        elapsed = round(time.time() - start, 2)
        st.success(f"Done in {elapsed}s")

        # =====================================================
        # Show LLM output
        # =====================================================
        st.subheader("LLM Output")
        st.write(answer)

        # =====================================================
        # Show top relevant chunks
        # =====================================================
        st.subheader("Top Relevant Chunks")
        for i, chunk in enumerate(chunks, start=1):
            with st.expander(f"Chunk #{i}"):
                st.write(chunk['text'])

        # =====================================================
        # Show compressed context
        # =====================================================
        with st.expander("Compressed Context Sent to LLM"):
            for i, c in enumerate(compressed, start=1):
                st.markdown(f"{c['text']}")
                st.markdown("---")