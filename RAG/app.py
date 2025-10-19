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

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="wide")
inject_css()

client = genai.Client(api_key=GOOGLE_API_KEY)
dataset = load_dataset()
vectordb = build_vectorstore(dataset)

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej hej! 👋 Welcome to Sweden. What would you like to explore today?"}
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "last_location" not in st.session_state:
    st.session_state.last_location = None  # store lat/lon/city for follow-up


# ----------------- HEADER -----------------
st.markdown("""
<div class="header-block">
    <div class="main-title">🇸🇪 GuideMe Sweden 🇸🇪</div>
    <div class="sub-title">Explore Sweden with your smart travel companion — powered by Gemini ✨</div>
</div>
""", unsafe_allow_html=True)


# ----------------- SIDEBAR -----------------
st.sidebar.markdown("## 🧭 GuideMe Tools")

uploaded_file = st.sidebar.file_uploader(
    "📸 Upload an image (optional)",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

if uploaded_file:
    st.session_state.uploaded_image = uploaded_file
    try:
        st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
    except TypeError:
        st.sidebar.image(uploaded_file, caption=uploaded_file.name, width=300)

    if st.sidebar.button("Remove image"):
        st.session_state.uploaded_image = None
        st.rerun()

show_debug = st.sidebar.checkbox("🔧 Show RAG Debug Info", value=False)
st.sidebar.markdown("---")
st.sidebar.caption("Upload a photo of a place or landmark 🏰 — I’ll try to identify it for you!")


# ----------------- CHAT DISPLAY -----------------
st.markdown('<div class="page">', unsafe_allow_html=True)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])


# ----------------- CHAT INPUT -----------------
user_query = st.chat_input("Ask something about Sweden...")


# ===================================================
# ============= RESTAURANT FOLLOW-UP LOGIC ==========
# ===================================================
if user_query and user_query.lower().strip() in [
    "yes", "sure", "ok", "okay", "please do", "yes please", "show me restaurants"
]:
    last_loc = st.session_state.get("last_location")
    if last_loc:
        nearby = find_nearby_places(dataset, last_loc["lat"], last_loc["lon"], 20)
        restaurants = [r for r in nearby if r.get("category") == "FoodEstablishment"]

        if restaurants:
            st.markdown(f"### 🍴 Top Restaurants Near {last_loc.get('city', 'Your Location')}")
            cols = st.columns(2)
            for i, r in enumerate(restaurants[:6]):
                with cols[i % 2]:
                    rating = (
                        f"⭐ {r.get('rating', '?')}/5 ({r.get('userRatingCount', '?')} reviews)"
                        if r.get("rating") else ""
                    )
                    maps_link = r.get("googleMapsUri") or r.get("url") or ""
                    st.markdown(f"""
                    <div class="card">
                        <strong>{r.get('name')}</strong><br>
                        {rating}<br>
                        📍 {r.get('formattedAddress', '')}<br>
                        <a href="{maps_link}" target="_blank">Open in Google Maps</a><br>
                        {r.get('description','')}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No restaurants found nearby. Try another location 🍽️")
        st.stop()


# ===================================================
# ================= NORMAL CHAT FLOW ================
# ===================================================
if user_query:
    image_to_send = st.session_state.uploaded_image
    has_image = image_to_send is not None

    # Add user message
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
            "Let’s keep our chat about Sweden 🌿 — maybe explore Stockholm’s old town or "
            "the northern lights in Kiruna?"
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

    # ------------- DETECT SUMMARIZATION REQUEST -------------
    q_lower = norm_q.lower()
    is_summary_request = any(word in q_lower for word in ["summarize", "summary", "in short", "overview", "short version"])

    # ------------- HYBRID PROMPT -------------
    if is_summary_request:
        hybrid_prompt = f"""
        You are GuideMe Sweden, a concise and clear Swedish travel expert.

        ### Task:
        Summarize the relevant information about the place or topic mentioned below.
        - Provide a short, **3–4 sentence** summary.
        - Focus on key highlights and cultural or historical significance.
        - Avoid repetition or unnecessary details.
        - Maintain a warm, travel-guide tone.

        ### Context:
        {context}

        ### Input:
        {norm_q}
        """
    else:
        hybrid_prompt = f"""
        You are GuideMe Sweden, a warm, friendly and **engaging** Swedish travel companion.

        ### Instructions:
        - Always respond in **English**, preserving Swedish names (Göteborg, Västra Götaland, etc.).
        - Be empathetic, enthusiastic, and conversational like a real travel guide.
        - Use context if relevant, and feel free to add brief cultural insights or travel facts.
        - Never invent details — rely on verified Swedish data or retrieved context.

        ### Knowledge:
        You have access to a Swedish tourism dataset retrieved from a vector database.
        This includes:
        - Landmarks, cities, museums, hotels, and nature attractions.
        - Restaurants and food establishments (category: "FoodEstablishment") with location coordinates.

        ### Context:
        {context}

        ### Conversation so far:
        {[m['content'] for m in st.session_state.messages[-3:]]}

        ### Question:
        {norm_q}
        """

    # ------------- GEMINI MULTIMODAL INPUT -------------
    contents = [hybrid_prompt]
    if has_image:
        image = Image.open(image_to_send)
        contents.append(image)

    # ------------- STREAMING RESPONSE -------------
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

        # Skip restaurant suggestions if summarizing
        if docs and not is_summary_request:
            top_meta = docs[0].metadata
            lat, lon = top_meta.get("latitude"), top_meta.get("longitude")
            city = top_meta.get("city")

            if lat and lon:
                st.session_state.last_location = {"lat": float(lat), "lon": float(lon), "city": city}

                nearby_restaurants = find_nearby_places(dataset, float(lat), float(lon), max_distance_km=10)
                restaurants = [r for r in nearby_restaurants if r.get("category") == "FoodEstablishment"]

                if restaurants:
                    suggestion_msg = (
                        f"I found some lovely restaurants near {city or 'this place'} 🍽️ — "
                        f"would you like me to show you a few top-rated options?"
                    )
                else:
                    suggestion_msg = (
                        f"I couldn’t find specific restaurants near {city or 'this area'}, "
                        f"but I can suggest popular Swedish dishes if you'd like ☕."
                    )

                st.session_state.messages.append({"role": "assistant", "content": suggestion_msg})
                render_bubble("assistant", suggestion_msg)

    except Exception as e:
        st.error(f"Gemini streaming failed: {e}")

    # Reset image session data
    st.session_state.uploaded_image = None
    st.session_state.uploaded_image_name = None
    st.rerun()
