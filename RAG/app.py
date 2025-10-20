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

st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="wide")
inject_css()

client = genai.Client(api_key=GOOGLE_API_KEY)
dataset = load_dataset()
vectordb = build_vectorstore(dataset)

# load friendly Q&A dataset
try:
    with open("qa.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    # st.sidebar.success("QA dataset loaded successfully ✅")
except Exception as e:
    qa_pairs = []
    st.sidebar.error(f"Could not load QA dataset: {e}")

# load restaurants data from separate json
try:
    with open("ratings_food.json", "r", encoding="utf-8") as f:
        restaurant_ratings = json.load(f)
    # st.sidebar.success("Restaurant ratings dataset loaded")
except Exception as e:
    restaurant_ratings = []
    st.sidebar.error(f"Could not load restaurant dataset: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej hej! 👋 Welcome to Sweden. What would you like to explore today?"}
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "last_location" not in st.session_state:
    st.session_state.last_location = None  # store lat/lon/city for follow-up

st.markdown("""
<div class="header-block">
    <div class="main-title">🇸🇪 GuideMe Sweden 🇸🇪</div>
    <div class="sub-title">Explore Sweden with your smart travel companion — powered by Gemini ✨</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🧭 GuideMe Tools")

# Initialize dynamic uploader key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded_file = st.sidebar.file_uploader(
    "📸 Upload an image (optional)",
    type=["jpg", "jpeg", "png"],
    key=f"image_uploader_{st.session_state.uploader_key}"
)

if uploaded_file:
    st.session_state.uploaded_image = uploaded_file
    try:
        st.sidebar.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
    except TypeError:
        st.sidebar.image(uploaded_file, caption=uploaded_file.name, width=300)

    if st.sidebar.button("Remove image"):
        st.session_state.uploaded_image = None
        st.session_state.uploader_key += 1  # force uploader to reset
        st.rerun()



# show_debug = st.sidebar.checkbox("🔧 Show RAG Debug Info", value=False)

show_debug = False
st.sidebar.markdown("---")
st.sidebar.caption("Upload a photo of a place or landmark 🏰 — I’ll try to identify it for you!")

# Chat display
st.markdown('<div class="page">', unsafe_allow_html=True)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])

user_query = st.chat_input("Ask something about Sweden...")

# Restaurant followup logic
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

# <---------------- MAIN CHAT FLOW ---------------->
if user_query:
    image_to_send = st.session_state.uploaded_image
    has_image = image_to_send is not None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
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

    # build context
    context = "\n\n".join(
        f"{d.page_content}\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
        for d in docs
    )

    # Add friendly Q&A context if any matching question exists
    if qa_pairs:
        for qa in qa_pairs:
            if qa["question"].lower() in norm_q.lower():
                context += f"\n\nAdditional Q&A:\nQ: {qa['question']}\nA: {qa['answer']}"
                break



    # Detect if summarization or restaurant question
    q_lower = norm_q.lower()
    is_summary_request = any(word in q_lower for word in ["summarize", "summary", "overview", "short version"])
    is_food_query = any(word in q_lower for word in [
        "restaurant", "food", "eat", "cafe", "lunch", "dinner", "pizza", "burger", "dining"
    ])

    # Detect location name from docs or user query
    place_name = None
    if docs:
        meta = docs[0].metadata
        place_name = meta.get("city") or meta.get("region") or meta.get("name")

    if not place_name:
        for w in user_query.split():
            if w.istitle() and len(w) > 3:
                place_name = w
                break

    # restaurant rating context
    restaurant_context = ""
    top_rated = []
    if is_food_query and isinstance(restaurant_ratings, list):
        matched = [r for r in restaurant_ratings if isinstance(r, dict)]
        if place_name:
            matched = [
                r for r in matched
                if place_name.lower() in (r.get("formattedAddress", "") + r.get("name", "")).lower()
            ]
        top_rated = sorted(matched, key=lambda x: x.get("rating") or 0, reverse=True)[:6]

    else:
        top_rated = []

    if top_rated:
        restaurant_context = "\n\n".join([
            f"{r['name']} — Rated {r.get('rating','?')}/5 "
            f"({r.get('userRatingCount','?')} reviews). "
            f"Located at {r.get('formattedAddress','N/A')}. "
            f"Google Maps: {r.get('googleMapsUri','')}"
            for r in top_rated
        ])
        if show_debug:
            st.sidebar.success(f"Found {len(top_rated)} matching restaurants for {place_name}")

    
    if is_summary_request:
        hybrid_prompt = f"""
        You are GuideMe Sweden, a concise and clear Swedish travel expert.

        ### Task:
        Summarize the relevant information about the place or topic mentioned below.
        - Always respond in **English**, preserving Swedish names (Göteborg, Västra Götaland, etc.).
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
        - Use context if relevant, and include real restaurant data when available.
        - Never invent details — rely on verified Swedish data or retrieved context.
        - If a relevant question or topic exists in the **Q&A dataset (qa.json)**, prefer using that verified answer in your response, while keeping a natural tone.


        ### Knowledge:
        You have access to:
        - A Swedish tourism dataset (from ChromaDB)
        - Restaurant ratings (from Google Maps JSON)
        - Common questions and answers about Swedish tradition and its culture (qa.json)

        ### Context:
        {context}

        ### Restaurant Ratings (if relevant):
        {restaurant_context if is_food_query else "No restaurant rating data relevant for this question."}

        ### Conversation so far:
        {[m['content'] for m in st.session_state.messages[-3:]]}

        ### Question:
        {norm_q}
        """

    # gemini multimodal call
    contents = [hybrid_prompt]
    if has_image:
        image = Image.open(image_to_send)
        contents.append(image)

    # streaming
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

        # (Optional) Display restaurant cards visually
        if top_rated:
            st.markdown("### 🍽️ Top Rated Restaurants")
            cols = st.columns(2)
            for i, r in enumerate(top_rated):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                        <strong>{r['name']}</strong><br>
                        ⭐ {r.get('rating','?')}/5 ({r.get('userRatingCount','?')} reviews)<br>
                        📍 {r.get('formattedAddress','')}<br>
                        <a href="{r.get('googleMapsUri','')}" target="_blank">Open in Google Maps</a>
                    </div>
                    """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Gemini streaming failed: {e}")

    # Reset uploaded image
    st.session_state.uploaded_image = None
    st.session_state.uploaded_image_name = None
    st.rerun()
