from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# Olist
# ---------------------------------------------------------

OLIST_DATA_DIR = ROOT / "data" / "olist"
OLIST_DB_PATH = OLIST_DATA_DIR / "olist.sqlite"

# Keep Olist embeddings completely separate from Spider.
OLIST_PERSIST_DIR = str(ROOT / "olist_chroma_db")