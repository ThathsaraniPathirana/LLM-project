import streamlit as st, time, json, asyncio
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
from utils.mcp_utils import fetch_places
from PIL import Image

st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="wide")
inject_css()

client = genai.Client(api_key=GOOGLE_API_KEY)
dataset = load_dataset()
vectordb = build_vectorstore(dataset)

# Load friendly Q&A dataset
try:
    with open("qa.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
except Exception as e:
    qa_pairs = []
    st.sidebar.error(f"Could not load QA dataset: {e}")

# Load restaurants data from separate json
try:
    with open("ratings_food.json", "r", encoding="utf-8") as f:
        restaurant_ratings = json.load(f)
except Exception as e:
    restaurant_ratings = []
    st.sidebar.error(f"Could not load restaurant dataset: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hej hej! 👋 Welcome to Sweden. What would you like to explore today?"}
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "last_location" not in st.session_state:
    st.session_state.last_location = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# NEW: MCP-related state
if "pending_mcp_request" not in st.session_state:
    st.session_state.pending_mcp_request = None

if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "waiting_for_location": False,
        "intent": None,  # "restaurant" or "hotel"
        "location": None,
        "original_query": None
    }

if "use_live_data" not in st.session_state:
    st.session_state.use_live_data = False

st.markdown("""
<div class="header-block">
    <div class="main-title">🇸🇪 GuideMe Sweden 🇸🇪</div>
    <div class="sub-title">Explore Sweden with your smart travel companion — powered by Gemini ✨</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🧭 GuideMe Tools")

# MCP Live Data Toggle
st.session_state.use_live_data = st.sidebar.toggle(
    "🔴 Use Live Data (Google Places API)",
    value=st.session_state.use_live_data,
    help="Fetch real-time restaurant and hotel recommendations"
)

if st.session_state.use_live_data:
    st.sidebar.info("✨ Live mode: Will ask before fetching from Google Places")
else:
    st.sidebar.info("📚 Dataset mode: Using cached data")

# Image uploader
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
        st.session_state.uploader_key += 1
        st.rerun()

show_debug = st.sidebar.checkbox("🔧 Show Debug Info", value=False)

if show_debug:
    st.sidebar.markdown("###  Conversation Context")
    st.sidebar.json(st.session_state.conversation_context)
    if st.session_state.pending_mcp_request:
        st.sidebar.markdown("### Pending MCP Request")
        st.sidebar.json(st.session_state.pending_mcp_request)

st.sidebar.markdown("---")
st.sidebar.caption("Upload a photo of a place or landmark 🏰 — I'll try to identify it for you!")

# Chat display
st.markdown('<div class="page">', unsafe_allow_html=True)
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])

# Helper functions for MCP
def detect_query_intent(query: str) -> dict:
    """Detect if query is asking for restaurants or hotels."""
    q_lower = query.lower()
    
    is_food_query = any(word in q_lower for word in [
        "restaurant", "food", "eat", "cafe", "lunch", "dinner", 
        "pizza", "burger", "dining", "where to eat", "good place to eat",
        "good restaurant", "best restaurant"
    ])
    
    is_hotel_query = any(word in q_lower for word in [
        "hotel", "stay", "accommodation", "lodge", "hostel", 
        "where to stay", "sleep", "place to stay", "best hotel"
    ])
    
    return {
        "is_food_query": is_food_query,
        "is_hotel_query": is_hotel_query,
        "intent": "restaurant" if is_food_query else ("hotel" if is_hotel_query else None)
    }

def extract_location_from_query(query: str) -> str:
    """Extract Swedish location from query."""
    swedish_locations = {
        "stockholm": "Stockholm", "gamla stan": "Gamla Stan, Stockholm",
        "södermalm": "Södermalm, Stockholm", "östermalm": "Östermalm, Stockholm",
        "vasastan": "Vasastan, Stockholm", "kungsholmen": "Kungsholmen, Stockholm",
        "gothenburg": "Gothenburg", "göteborg": "Gothenburg",
        "malmö": "Malmö", "malmo": "Malmö", "uppsala": "Uppsala",
        "västerås": "Västerås", "örebro": "Örebro", "linköping": "Linköping",
        "gävle": "Gävle", "gavle": "Gävle", "helsingborg": "Helsingborg",
        "jönköping": "Jönköping", "norrköping": "Norrköping", "lund": "Lund",
        "umeå": "Umeå", "umea": "Umeå", "borås": "Borås", "boras": "Borås",
        "eskilstuna": "Eskilstuna", "kiruna": "Kiruna", "visby": "Visby",
        "karlstad": "Karlstad", "växjö": "Växjö", "vaxjo": "Växjö"
    }
    
    q_lower = query.lower()
    
    # Check for exact matches (prioritize longer matches)
    sorted_locations = sorted(swedish_locations.items(), key=lambda x: len(x[0]), reverse=True)
    for key, value in sorted_locations:
        if key in q_lower:
            return value
    
    return None

# Handle pending MCP request
if st.session_state.pending_mcp_request:
    request = st.session_state.pending_mcp_request
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin: 10px 0; color: white;">
        <h4>🔴 Fetch Live Data?</h4>
        <p>I can get real-time <strong>{request['category']}</strong> recommendations near <strong>{request['location']}</strong> 
        from Google Places with current ratings and prices.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("✅ Yes, fetch live data", key="approve_mcp", use_container_width=True):
            with st.spinner(f"🔍 Fetching live {request['category']} data from Google Places..."):
                live_data = asyncio.run(fetch_places(
                    location=request["location"],
                    category=request["category"],
                    radius=2000,
                    max_results=5
                ))
                
                if live_data and live_data.get("places"):
                    response_text = f"### 🎯 Live {request['category'].title()} Recommendations\n\n"
                    response_text += f"Here are {len(live_data['places'])} top-rated options near {request['location']}:\n\n"
                    
                    for i, place in enumerate(live_data['places'], 1):
                        response_text += f"**{i}. {place['name']}**\n"
                        response_text += f"   - ⭐ {place['rating']}/5 ({place.get('total_ratings', 0):,} reviews)\n"
                        response_text += f"   - 💰 {place.get('price_level', 'Price not available')}\n"
                        response_text += f"   - 📍 {place['address']}\n"
                        response_text += f"   - [View on Google Maps]({place['maps_url']})\n\n"
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    error_msg = f"I couldn't fetch live data for {request['category']}s near {request['location']} right now. Let me show you what I have from my cached data instead."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                st.session_state.pending_mcp_request = None
                st.session_state.conversation_context = {
                    "waiting_for_location": False,
                    "intent": None,
                    "location": None,
                    "original_query": None
                }
                st.rerun()
    
    with col2:
        if st.button("📚 Use cached data", key="use_cached", use_container_width=True):
            cached_msg = "No problem! I'll use my cached restaurant data instead."
            st.session_state.messages.append({"role": "assistant", "content": cached_msg})
            st.session_state.pending_mcp_request = None
            st.session_state.conversation_context = {
                "waiting_for_location": False,
                "intent": None,
                "location": None,
                "original_query": None
            }
            st.rerun()
    
    with col3:
        if st.button("❌", key="decline_mcp", use_container_width=True):
            st.session_state.pending_mcp_request = None
            st.session_state.conversation_context = {
                "waiting_for_location": False,
                "intent": None,
                "location": None,
                "original_query": None
            }
            st.rerun()

user_query = st.chat_input("Ask something about Sweden...")

# Restaurant followup logic (existing feature preserved)
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

# Main chat flow
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
            "Let's keep our chat about Sweden 🌿 — maybe explore Stockholm's old town or "
            "the northern lights in Kiruna?"
        )
        st.session_state.messages.append({"role": "assistant", "content": safe_msg})
        render_bubble("assistant", safe_msg)
        st.session_state.uploaded_image = None
        st.rerun()

    # Normalize query
    norm_q = normalize_user_query_spelling(user_query)
    
    # Check if we're waiting for location clarification
    if st.session_state.conversation_context.get("waiting_for_location"):
        location = extract_location_from_query(user_query)
        
        if location:
            # Got the location!
            ctx = st.session_state.conversation_context
            
            if st.session_state.use_live_data:
                # Trigger MCP consent flow
                st.session_state.pending_mcp_request = {
                    "location": location,
                    "category": ctx["intent"],
                    "original_query": ctx["original_query"]
                }
                st.session_state.conversation_context = {
                    "waiting_for_location": False,
                    "intent": None,
                    "location": None,
                    "original_query": None
                }
                st.rerun()
            else:
                # Use cached data - continue with normal flow
                acknowledge_msg = f"Got it! Let me find {ctx['intent']}s in {location} from my database..."
                st.session_state.messages.append({"role": "assistant", "content": acknowledge_msg})
                render_bubble("assistant", acknowledge_msg)
                st.session_state.conversation_context = {
                    "waiting_for_location": False,
                    "intent": None,
                    "location": None,
                    "original_query": None
                }
                # Continue with normal flow below
        else:
            # Still no clear location
            clarify_msg = "I didn't catch the city name. Could you mention a Swedish city? For example: Stockholm, Gothenburg, Malmö, Uppsala, or Gävle?"
            st.session_state.messages.append({"role": "assistant", "content": clarify_msg})
            render_bubble("assistant", clarify_msg)
            st.stop()
    
    # Detect intent in current query
    intent_data = detect_query_intent(user_query)
    location = extract_location_from_query(user_query)
    
    if show_debug:
        st.sidebar.write("🎯Intent Detection:")
        st.sidebar.json({
            "intent": intent_data.get("intent"),
            "location": location,
            "live_mode": st.session_state.use_live_data
        })
    
    # If user wants restaurants/hotels but didn't specify location
    if intent_data.get("intent") and not location and st.session_state.use_live_data:
        # Save context and ask for location
        st.session_state.conversation_context = {
            "waiting_for_location": True,
            "intent": intent_data["intent"],
            "location": None,
            "original_query": user_query
        }
        
        # Generate natural location question
        if intent_data["intent"] == "restaurant":
            ask_location_msg = "I'd love to help you find great restaurants! 🍽️ Which city or area in Sweden are you interested in?"
        else:
            ask_location_msg = "I can help you find excellent hotels! 🏨 Which city in Sweden would you like to stay in?"
        
        st.session_state.messages.append({"role": "assistant", "content": ask_location_msg})
        render_bubble("assistant", ask_location_msg)
        st.stop()
    
    # If we have both intent and location and live mode is ON
    if intent_data.get("intent") and location and st.session_state.use_live_data:
        # Trigger MCP consent flow
        st.session_state.pending_mcp_request = {
            "location": location,
            "category": intent_data["intent"],
            "original_query": user_query
        }
        st.rerun()

    # Regular RAG flow
    docs = vectordb.similarity_search(norm_q, k=TOP_K)

    if show_debug:
        st.sidebar.write(f"🔎 Retrieved {len(docs)} documents")
        with st.sidebar.expander("Retrieved Context", expanded=False):
            st.code(
                "\n\n".join(
                    f"{d.page_content[:400]}...\nMeta:{json.dumps(d.metadata, ensure_ascii=False)}"
                    for d in docs
                ) or "No documents retrieved."
            )

    # Build context
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

    # Detect query types
    q_lower = norm_q.lower()
    is_summary_request = any(word in q_lower for word in [
        "summarize", "summary", "overview", "short version"
    ])
    is_food_query = intent_data.get("is_food_query")

    # Detect location name from docs or user query
    place_name = location
    if not place_name and docs:
        meta = docs[0].metadata
        place_name = meta.get("city") or meta.get("region") or meta.get("name")

    if not place_name:
        for w in user_query.split():
            if w.istitle() and len(w) > 3:
                place_name = w
                break

    # Restaurant rating context from cached JSON
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

        if top_rated:
            restaurant_context = "\n\n".join([
                f"{r['name']} — Rated {r.get('rating','?')}/5 "
                f"({r.get('userRatingCount','?')} reviews). "
                f"Located at {r.get('formattedAddress','N/A')}. "
                f"Google Maps: {r.get('googleMapsUri','')}"
                for r in top_rated
            ])

    # Build prompt
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
        live_note = ""
        if st.session_state.use_live_data:
            live_note = "\n**Note**: Live data mode is enabled. If user asks about restaurants/hotels without location, I've already asked them for clarification."
        
        hybrid_prompt = f"""
You are GuideMe Sweden, a warm, friendly and **engaging** Swedish travel companion.

### Instructions:
- Always respond in **English**, preserving Swedish names (Göteborg, Västra Götaland, etc.).
- Be empathetic, enthusiastic, and conversational like a real travel guide.
- Use context if relevant, and include real restaurant data when available.
- Never invent details — rely on verified Swedish data or retrieved context.
- If a relevant question exists in the Q&A dataset, prefer that verified answer.
{live_note}

### Knowledge:
You have access to:
- Swedish tourism dataset (ChromaDB)
- Restaurant ratings (cached Google Maps JSON)
- Common Q&A about Swedish culture (qa.json)
- Live Google Places data (when user approves)

### Context:
{context}

### Restaurant Data (cached):
{restaurant_context if is_food_query else "No cached restaurant data relevant."}

### Recent Conversation:
{[m['content'] for m in st.session_state.messages[-3:]]}

### Question:
{norm_q}
"""

    # Gemini multimodal call
    contents = [hybrid_prompt]
    if has_image:
        image = Image.open(image_to_send)
        contents.append(image)

    # Streaming response
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

        # Display cached restaurant cards if available
        if top_rated:
            st.markdown("### 🍽️ Top Rated Restaurants (Cached Data)")
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
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)