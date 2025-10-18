import os, json, re, math, time
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv
from google import genai  # ✅ new Gemini SDK
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from PIL import Image

# ---------------- CONFIG ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PERSIST_DIR = "./chroma_db"
DATA_PATH = "./final_dataset.json"
EMBED_MODEL = "text-embedding-004"
TOP_K = 6
RADIUS_KM = 20

# ---------------- COLORS ----------------
NAVY = "#001B44"
GOLD = "#FFD43B"
WHITE = "#F8FAFC"
TEXT_LIGHT = "#E6E6E6"

# ---------------- UTILITIES ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearby_places(dataset, base_lat, base_lon, max_distance_km=RADIUS_KM):
    results = []
    for rec in dataset:
        lat, lon = rec.get("latitude"), rec.get("longitude")
        if lat and lon:
            try:
                dist = haversine(float(base_lat), float(base_lon), float(lat), float(lon))
                if dist <= max_distance_km:
                    rec["distance_km"] = round(dist, 1)
                    results.append(rec)
            except Exception:
                pass
    results.sort(key=lambda x: x.get("distance_km", 999))
    return results[:8]

def detect_nearby_query(text):
    return any(w in text.lower() for w in ["nearby","close","around","near","next to","surrounding"])

INPUT_ALIASES = {
    "gotenburg": "Göteborg","gothenburg":"Göteborg","goteborg":"Göteborg",
    "linkoping":"Linköping","ostergotland":"Östergötland",
    "vastra gotaland":"Västra Götaland","varmland":"Värmland",
    "orebro":"Örebro","gavle":"Gävle","angelholm":"Ängelholm",
}
NAME_REMAP = {
    r"\bGothenburg\b": "Göteborg", r"\bOrebro\b": "Örebro",
    r"\bOstergotland\b": "Östergötland", r"\bVastra Gotaland\b": "Västra Götaland",
    r"\bVarmland\b": "Värmland", r"\bGavle\b": "Gävle", r"\bAngelholm\b": "Ängelholm",
}
TYPE_FIELD_MAP = {
    "schema:Event":["name","description","facts_text","start_date","region","city","location_name"],
    "schema:LodgingBusiness":["name","description","facts_text","city","street","checkin_time","checkout_time","region"],
    "schema:Place":["name","description","facts_text","region","city","street"],
    "schema:Trip":["name","description","facts_text","region","city"],
    "schema:FoodEstablishment":["name","description","facts_text","region","city","street"],
    "_default":["name","description","facts_text","region","city","street"],
}

def _lower_ascii(s): 
    return s.lower().replace("å","a").replace("ä","a").replace("ö","o").replace("é","e")

def normalize_user_query_spelling(q):
    q_norm=q; q_lc=_lower_ascii(q)
    for bad,good in INPUT_ALIASES.items():
        if bad in q_lc: q_norm=q_norm.replace(bad,good)
    return q_norm

def load_dataset(p):
    with open(p,"r",encoding="utf-8") as f: data=json.load(f)
    return data if isinstance(data,list) else []

def map_fields_by_type(rec): 
    return TYPE_FIELD_MAP.get(rec.get("type") or rec.get("@type") or "", TYPE_FIELD_MAP["_default"])

def to_map_link(lat,lon): 
    return f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else None

def extract_meta(r):
    img=r.get("main_image") or r.get("image")
    lat,lon=r.get("latitude"),r.get("longitude")
    return dict(name=r.get("name") or r.get("alternate_name") or "Unnamed",
                city=r.get("city"), region=r.get("region"),
                url=r.get("url"), image=img, map_link=to_map_link(lat,lon),
                latitude=lat, longitude=lon)

def make_doc_from_record(r):
    fields=map_fields_by_type(r)
    lines=[f"{f}: {r.get(f)}" for f in fields if r.get(f)]
    return Document(page_content="\n".join(lines), metadata=extract_meta(r))

def build_or_load_vectorstore(records,embeddings):
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    docs=[make_doc_from_record(r) for r in records]
    db=Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
    db.persist(); return db

def preserve_swedish_names(t):
    for pat,swe in NAME_REMAP.items(): t=re.sub(pat,swe,t)
    return t

def is_safe_input(text: str) -> bool:
    banned = ["sex","suicide","kill","weapon","hate","politics","religion","terrorism","drugs"]
    return not any(b in text.lower() for b in banned)

def sanitize_output(text: str) -> str:
    banned = ["kill","hate","suicide","weapon","drugs","terrorism"]
    if any(b in text.lower() for b in banned):
        return "I’m sorry, I can’t discuss that. Let’s talk about Sweden instead!"
    return text

# ---------------- UI ----------------
st.set_page_config(page_title="GuideMe Sweden", page_icon="🇸🇪", layout="centered")

st.markdown(f"""
<style>
.stApp {{
  background-color: {NAVY};
  color: {TEXT_LIGHT};
  font-family: 'Inter', sans-serif;
}}
.header {{
  text-align: center;
  background: linear-gradient(90deg, {NAVY} 30%, {GOLD} 30%, {GOLD} 35%, {NAVY} 35%);
  padding: 0.6em;
  color: {GOLD};
  font-weight: 900;
  font-size: 1.8em;
}}
.card {{
  background-color: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 10px;
  padding: 10px;
  margin: 10px 0;
}}
.user-bubble {{
  background-color: {GOLD};
  color: black;
  padding: 0.8em 1em;
  border-radius: 1em;
  margin: 0.6em auto;
  max-width: 80%;
}}
.bot-bubble {{
  background-color: rgba(255,255,255,0.1);
  color: {WHITE};
  padding: 0.8em 1em;
  border-radius: 1em;
  margin: 0.6em auto;
  max-width: 80%;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">GuideMe Sweden</div>', unsafe_allow_html=True)
st.caption("Explore Sweden with your smart travel companion — powered by Gemini ✨")

# ---------------- MODEL INIT ----------------
if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY in .env file.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
dataset = load_dataset(DATA_PATH)
vectordb = build_or_load_vectorstore(dataset, embeddings)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = st.session_state.get("chat_history", [])


uploaded_image = st.file_uploader("📸 Upload an image (optional):", type=["jpg", "png", "jpeg"])
user_query = st.chat_input("Ask something about Sweden...")

# ---------------- IMAGE HANDLER ----------------

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded image", use_container_width=True)
    with st.spinner("Analyzing image..."):
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["Identify this place in Sweden. If unknown, say 'Sorry, I don’t know this place.'", image]
            )
            msg = f"📍 {res.text.strip()}"
            st.session_state.chat_history.append(("assistant", msg))
            st.markdown(f'<div class="bot-bubble">{msg}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Image analysis failed: {e}")

# ---------------- CHAT HANDLER ----------------
if user_query:
    normalized_query = normalize_user_query_spelling(user_query)
    st.session_state.chat_history.append(("user", user_query))
    st.markdown(f'<div class="user-bubble">{user_query}</div>', unsafe_allow_html=True)

    if not is_safe_input(user_query):
        msg = "Let’s keep our chat about Sweden — maybe you’d like to explore Stockholm’s old town or see the northern lights in Kiruna?"
        st.session_state.chat_history.append(("assistant", msg))
        st.warning(msg)
        st.stop()

    docs = vectordb.similarity_search(normalized_query, k=TOP_K)
    context_block = "\n\n".join(
        f"{d.page_content}\nMeta: {json.dumps(d.metadata, ensure_ascii=False)}"
        for d in docs
    )

    nearby_context, nearby_cards = "", []
    if detect_nearby_query(user_query) and docs:
        ref = docs[0].metadata
        lat, lon = ref.get("latitude"), ref.get("longitude")
        if lat and lon:
            nearby = find_nearby_places(dataset, float(lat), float(lon))
            for n in nearby:
                nearby_cards.append(n)
            nearby_context = "\n".join(
                f"{n.get('name')} ({n.get('category')}, {n.get('distance_km','?')} km away)"
                for n in nearby
            )

    hybrid_prompt = f"""
You are GuideMe Sweden, a warm, friendly and **engaging** Swedish travel companion.

### Instructions:
- Always respond in **English**, preserving Swedish names (Göteborg, Västra Götaland, etc.).
- Be enthusiastic, empathetic, and conversational like a real travel guide.
- Use context if relevant, but add small facts or fun notes from your knowledge if missing.
- You don’t always need to start with "Hej!" — vary your style naturally.
- Avoid generic filler. Give warm, human-like, slightly playful answers.
- Be factual, concise, and sound genuinely interested in helping travelers.

### Context:
{context_block}

### Nearby info:
{nearby_context}

### Question:
{normalized_query}
"""

    try:
        full_text = ""
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[hybrid_prompt]
        )

        placeholder = st.empty()
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
                clean = sanitize_output(full_text)
                if "Resources" in clean and not re.search(r"https?://", clean):
                    clean = re.sub(r"Resources:.*", "", clean, flags=re.S)
                placeholder.markdown(
                    f'<div class="bot-bubble">{preserve_swedish_names(clean.strip())}</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.05)

        st.session_state.chat_history.append(("assistant", preserve_swedish_names(full_text.strip())))

        if nearby_cards:
            st.markdown("### 🗺️ Nearby Places")
            cols = st.columns(2)
            for i, n in enumerate(nearby_cards):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                    <strong>{n.get('name')}</strong><br>
                    <span style='color:{GOLD}'>{n.get('category','')}</span><br>
                    {n.get('city','')}<br>
                    {n.get('distance_km','?')} km away<br>
                    <a href="{n.get('url') or n.get('map_link','')}">Visit</a>
                    </div>""", unsafe_allow_html=True)
                    if n.get("image"):
                        st.image(n["image"], use_container_width=True)

    except Exception as e:
        st.error(f"Gemini streaming failed: {e}")

# ---------------- INLINE HISTORY DISPLAY (No Expander) ----------------
# Skip last assistant message if it was already streamed
history = st.session_state.chat_history[:-1] if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "assistant" else st.session_state.chat_history

for role, msg in history:
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"