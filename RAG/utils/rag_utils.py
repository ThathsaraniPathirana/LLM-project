import os, json, math, re
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from config import PERSIST_DIR, DATA_PATH, EMBED_MODEL, RADIUS_KM

# type field map
TYPE_FIELD_MAP = {
    "schema:LodgingBusiness": [
        "name", "description", "facts_text", "city",
        "street", "checkin_time", "checkout_time", "region"
    ],
    "schema:Place": [
        "name", "description", "facts_text", "region", "city", "street"
    ],
    "schema:Trip": [
        "name", "description", "facts_text", "region", "city"
    ],
    "schema:FoodEstablishment": [
        "name", "description", "facts_text", "region", "city", "street"
    ],
    "_default": [
        "name", "description", "facts_text", "region", "city", "street"
    ],
}

def detect_nearby_query(text: str) -> bool:
    """Detect whether the user asked for something 'nearby'."""
    return any(w in text.lower() for w in ["nearby", "close", "around", "near", "next to", "surrounding"])


# text normalization
INPUT_ALIASES = {
    "gotenburg": "Göteborg", "gothenburg": "Göteborg", "goteborg": "Göteborg",
    "linkoping": "Linköping", "ostergotland": "Östergötland",
    "vastra gotaland": "Västra Götaland", "varmland": "Värmland",
    "orebro": "Örebro", "gavle": "Gävle", "angelholm": "Ängelholm",
}

NAME_REMAP = {
    r"\bGothenburg\b": "Göteborg", r"\bOrebro\b": "Örebro",
    r"\bOstergotland\b": "Östergötland", r"\bVastra Gotaland\b": "Västra Götaland",
    r"\bVarmland\b": "Värmland", r"\bGavle\b": "Gävle", r"\bAngelholm\b": "Ängelholm",
}


def _lower_ascii(s: str) -> str:
    """Normalize diacritics for Swedish names to ASCII before searching."""
    return s.lower().replace("å", "a").replace("ä", "a").replace("ö", "o").replace("é", "e")


def normalize_user_query_spelling(q: str) -> str:
    """Replace common English spellings with proper Swedish ones."""
    q_norm = q
    q_lc = _lower_ascii(q)
    for bad, good in INPUT_ALIASES.items():
        if bad in q_lc:
            q_norm = q_norm.replace(bad, good)
    return q_norm


def normalize_geo_terms(text: str) -> str:
    """Bridge Swedish–English mismatches (e.g. 'Lake Bysjön' → 'Bysjön')."""
    text = text.replace("lake ", "").replace("sjön", "sjö")
    return text


# data loading
def load_dataset():
    """Load main tourism dataset from JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


# document creation
def map_fields_by_type(rec):
    """Return the list of fields to use for a given schema type."""
    return TYPE_FIELD_MAP.get(rec.get("type") or rec.get("@type") or "", TYPE_FIELD_MAP["_default"])


def to_map_link(lat, lon):
    """Generate a clickable Google Maps link."""
    return f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else None


def extract_meta(r):
    """Extract metadata and ensure Chroma-safe primitive values."""
    img = r.get("main_image") or r.get("image")
    lat, lon = r.get("latitude"), r.get("longitude")

    meta = {
        "name": r.get("name") or r.get("alternate_name") or "Unnamed",
        "city": r.get("city"),
        "region": r.get("region"),
        "url": r.get("url"),
        "image": img,
        "map_link": to_map_link(lat, lon),
        "latitude": lat,
        "longitude": lon,
    }

    # Sanitize metadata — remove None, ensure primitives only
    safe_meta = {}
    for k, v in meta.items():
        if v is None:
            continue  
        if isinstance(v, (str, int, float, bool)):
            safe_meta[k] = v
        else:
            safe_meta[k] = str(v)

    return safe_meta



def make_doc_from_record(r):
    """Convert a single dataset record into a LangChain Document for embedding."""
    fields = map_fields_by_type(r)
    lines = [f"{f}: {r.get(f)}" for f in fields if r.get(f)]
    return Document(page_content="\n".join(lines), metadata=extract_meta(r))


# vector store build
def build_vectorstore(dataset):
    """Build or load the Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    # If already built, reuse persisted DB
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Otherwise, create from dataset
    docs = [make_doc_from_record(r) for r in dataset]
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
    db.persist()
    return db


# output utilities
def preserve_swedish_names(t: str) -> str:
    """Restore correct Swedish spellings after model output."""
    for pat, swe in NAME_REMAP.items():
        t = re.sub(pat, swe, t)
    return t


def is_safe_input(text: str) -> bool:
    """Prevent unsafe or off-topic user queries."""
    banned = ["sex", "suicide", "kill", "weapon", "hate", "politics", "religion", "terrorism", "drugs"]
    return not any(b in text.lower() for b in banned)


def sanitize_output(text: str) -> str:
    """Filter sensitive topics in model responses."""
    banned = ["kill", "hate", "suicide", "weapon", "drugs", "terrorism"]
    if any(b in text.lower() for b in banned):
        return "I’m sorry, I can’t discuss that. Let’s talk about Sweden instead!"
    return text
