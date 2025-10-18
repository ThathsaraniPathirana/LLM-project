import streamlit as st, time, json
from google import genai
from config import *
from utils.geo_utils import find_nearby_places
from utils.text_utils import (
    normalize_user_query_spelling,
    sanitize_output,
    preserve_swedish_names,
    is_safe_input
)
from utils.rag_utils import load_dataset, build_vectorstore
from utils.ui_utils import inject_css, render_bubble
from PIL import Image

# Page configuration
st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="wide")
inject_css()

client = genai.Client(api_key=GOOGLE_API_KEY)
dataset = load_dataset()
vectordb = build_vectorstore(dataset)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej hej! 👋 Welcome to Sweden. What would you like to explore today?"}
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "last_location" not in st.session_state:
    st.session_state.last_location = None  # store lat/lon/city for follow-up

# header
st.markdown("""
<div class="header-block">
    <div class="main-title">🇸🇪 GuideMe Sweden 🇸🇪</div>
    <div class="sub-title">Explore Sweden with your smart travel companion — powered by Gemini ✨</div>
</div>
""", unsafe_allow_html=True)

# Sidebar for image uploading
st.sidebar.markdown("## 🧭 GuideMe Tools")

uploaded_file = st.sidebar.file_uploader(
    "📸 Upload an image (optional)",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

if uploaded_file:
    st.session_state.uploaded_image = uploaded_file
    st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
    if st.sidebar.button("Remove image"):
        st.session_state.uploaded_image = None
        st.rerun()

show_debug = st.sidebar.checkbox("🔧 Show RAG Debug Info", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Upload a photo of a place or landmark 🏰 — I’ll try to identify it for you!")

# chat display
st.markdown('<div class="page">', unsafe_allow_html=True)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])

# chat input
user_query = st.chat_input("Ask something about Sweden...")

# <---------------Main chat logic------------->

# Handle follow-up "yes" or "please show" for restaurants
if user_query and user_query.lower().strip() in ["yes", "sure", "ok", "okay", "please do", "yes please", "show me restaurants"]:
    last_loc = st.session_state.get("last_location")
    if last_loc:
        nearby = find_nearby_places(dataset, last_loc["lat"], last_loc["lon"], 20)
        restaurants = [r for r in nearby if r.get("category") == "FoodEstablishment"]
        if restaurants:
            st.markdown("### 🍴 Recommended Restaurants Nearby")
            cols = st.columns(2)
            for i, r in enumerate(restaurants[:6]):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                    <strong>{r.get('name')}</strong><br>
                    <span style='color:{GOLD}'>{r.get('city','')}</span><br>
                    {r.get('description','')}<br>
                    <a href="{r.get('url') or ''}">Visit Website</a><br>
                    📍 {r.get('distance_km','?')} km away
                    </div>
                    """, unsafe_allow_html=True)
                    if r.get("image"):
                        st.image(r["image"], width=250)
        else:
            st.info("No restaurants found nearby. Try another location 🍽️")
        st.stop()

# normal chat flow
if user_query:
    image_to_send = st.session_state.uploaded_image
    has_image = image_to_send is not None

    # user query
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })
    render_bubble("user", user_query)
    if has_image:
        st.image(image_to_send, caption="Uploaded image", width=400)

    # Safety check
    if not is_safe_input(user_query):
        safe_msg = (
            "Let’s keep our chat about Sweden — maybe explore Stockholm’s old town or "
            "the northern lights in Kiruna? 🌿"
        )
        st.session_state.messages.append({"role": "assistant", "content": safe_msg})
        render_bubble("assistant", safe_msg)
        st.session_state.uploaded_image = None
        st.rerun()

    # RAG context fetching 
    norm_q = normalize_user_query_spelling(user_query)
    docs = vectordb.similarity_search(norm_q, k=TOP_K)

    if show_debug:
        st.sidebar.write(f" Retrieved {len(docs)} documents from ChromaDB")
        with st.sidebar.expander("Retrieved Context (debug)", expanded=False):
            st.code(
                "\n\n".join(
                    f"{d.page_content[:400]}...\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
                    for d in docs
                ) or " No documents retrieved."
            )

    
    context = "\n\n".join(
        f"{d.page_content}\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
        for d in docs
    )

    hybrid_prompt = f"""
You are GuideMe Sweden, a warm, friendly and **engaging** Swedish travel companion.

### Instructions:
- Always respond in **English**, preserving Swedish names (Göteborg, Västra Götaland, etc.).
- Be empathetic, enthusiastic, and conversational like a real travel guide.
- Use context if relevant, and feel free to add brief cultural insights or travel facts.
- Never invent details — rely on verified Swedish data or retrieved context.

### Knowledge:
You have access to a Swedish tourism dataset retrieved from a vector database.
Always prioritize context-based information before general knowledge.

### Context:
{context}

### Conversation so far:
{[m['content'] for m in st.session_state.messages[-3:]]}

### Question:
{norm_q}
"""

    # GEMINI multimodal input
    contents = [hybrid_prompt]
    if has_image:
        image = Image.open(image_to_send)
        contents.append(image)

    # Streaming
    placeholder = st.empty()
    streamed = ""
    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents
        ):
            if hasattr(chunk, "text") and chunk.text:
                streamed += chunk.text
                clean = sanitize_output(streamed)
                placeholder.markdown(
                    f'<div class="bot-bubble">{preserve_swedish_names(clean.strip())}</div>',
                    unsafe_allow_html=True,
                )
                time.sleep(0.03)

        final = preserve_swedish_names(streamed.strip())
        st.session_state.messages.append({"role": "assistant", "content": final})

        # Extract location metadata (if available)
        if docs:
            top_meta = docs[0].metadata
            lat, lon = top_meta.get("latitude"), top_meta.get("longitude")
            city = top_meta.get("city")
            if lat and lon:
                st.session_state.last_location = {"lat": float(lat), "lon": float(lon), "city": city}
                suggestion_msg = f"Would you like me to suggest some nearby restaurants around {city or 'this place'}? 🍽️"
                st.session_state.messages.append({"role": "assistant", "content": suggestion_msg})
                render_bubble("assistant", suggestion_msg)

    except Exception as e:
        st.error(f"Gemini streaming failed: {e}")

    st.session_state.uploaded_image = None
    st.session_state.uploaded_image_name = None
    st.rerun()
