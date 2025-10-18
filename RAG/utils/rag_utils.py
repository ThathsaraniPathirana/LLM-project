import os, json
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from config import PERSIST_DIR, DATA_PATH, EMBED_MODEL

TYPE_FIELD_MAP = {
    "schema:LodgingBusiness":["name","description","facts_text","city","street","checkin_time","checkout_time","region"],
    "schema:Place":["name","description","facts_text","region","city","street"],
    "schema:Trip":["name","description","facts_text","region","city"],
    "schema:FoodEstablishment":["name","description","facts_text","region","city","street"],
    "_default":["name","description","facts_text","region","city","street"],
}

def load_dataset():
    with open(DATA_PATH,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data if isinstance(data,list) else []

def extract_meta(r):
    img=r.get("main_image") or r.get("image")
    lat,lon=r.get("latitude"),r.get("longitude")
    return dict(name=r.get("name") or "Unnamed", city=r.get("city"),
                region=r.get("region"), url=r.get("url"),
                image=img, latitude=lat, longitude=lon)

def make_doc_from_record(r):
    fields=TYPE_FIELD_MAP.get(r.get("type") or r.get("@type") or "", TYPE_FIELD_MAP["_default"])
    lines=[f"{f}: {r.get(f)}" for f in fields if r.get(f)]
    return Document(page_content="\n".join(lines), metadata=extract_meta(r))

def build_vectorstore(dataset):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    docs=[make_doc_from_record(r) for r in dataset]
    db=Chroma.from_documents(docs, embedding=embeddings, persist_directory=PERSIST_DIR)
    db.persist()
    return db
