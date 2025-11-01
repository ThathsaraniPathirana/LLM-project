import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PERSIST_DIR = "./chroma_db"
DATA_PATH = "../final_dataset.json"
EMBED_MODEL = "models/text-embedding-004"
TOP_K = 6
RADIUS_KM = 20

# COLORS
NAVY = "#001B44"
GOLD = "#FFD43B"
WHITE = "#F8FAFC"
TEXT_LIGHT = "#E6E6E6"
