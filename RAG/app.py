import streamlit as st, time, json
from google import genai
from config import *
from utils.geo_utils import find_nearby_places
from utils.text_utils import (
    normalize_user_query_spelling, sanitize_output,
    preserve_swedish_names, is_safe_input
)
from utils.rag_utils import load_dataset, build_vectorstore
from utils.ui_utils import inject_css, render_bubble

# --- INIT ---
st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="wide")
inject_css()

client = genai.Client(api_key=GOOGLE_API_KEY)
dataset = load_dataset()
vectordb = build_vectorstore(dataset)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej hej! 👋 Welcome to Sweden. What would you like to explore today?"}
    ]

st.markdown('<div class="page"><div class="header">GuideMe Sweden</div>', unsafe_allow_html=True)
st.caption("Explore Sweden with your smart travel companion — powered by Gemini ✨")

# --- Chat display ---
for msg in st.session_state.messages:
    render_bubble(msg["role"], msg["content"])

# --- Chat input ---
user_query = st.chat_input("Ask something about Sweden...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    render_bubble("user", user_query)

    if not is_safe_input(user_query):
        safe_msg = "Let’s keep our chat about Sweden 🌿 — maybe you’d like to explore Stockholm’s old town or the northern lights in Kiruna?"
        st.session_state.messages.append({"role": "assistant", "content": safe_msg})
        render_bubble("assistant", safe_msg)
        st.stop()

    norm_q = normalize_user_query_spelling(user_query)

    # --- Debugging Toggle ---
    show_debug = st.sidebar.checkbox("🔧 Show RAG Debug Info")

    # --- RAG Retrieval ---
    docs = vectordb.similarity_search(norm_q, k=TOP_K)
    context = "\n\n".join(
        f"{d.page_content}\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
        for d in docs
    )

    # --- Debug Info ---
    if show_debug:
        st.sidebar.write(f"🔎 Retrieved {len(docs)} documents from ChromaDB")
        with st.expander("🔍 Retrieved Context (debug)", expanded=False):
            st.code(
                "\n\n".join(
                    f"{d.page_content[:400]}...\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
                    for d in docs
                ) or "⚠️ No documents retrieved."
            )

    # --- Gemini Prompt ---
    hybrid_prompt = f"""
You are GuideMe Sweden, a friendly travel companion.

### Context:
{context}

### Question:
{norm_q}
"""

    # --- Streaming Response ---
    placeholder = st.empty()
    streamed = ""
    for chunk in client.models.generate_content_stream(model="gemini-2.5-flash", contents=[hybrid_prompt]):
        if hasattr(chunk, "text") and chunk.text:
            streamed += chunk.text
            clean = sanitize_output(streamed)
            placeholder.markdown(
                f'<div class="bot-bubble">{preserve_swedish_names(clean.strip())}</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.03)

    final = preserve_swedish_names(streamed.strip())
    st.session_state.messages.append({"role": "assistant", "content": final})

st.markdown('</div>', unsafe_allow_html=True)
