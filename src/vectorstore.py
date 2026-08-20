from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

from config import OLIST_PERSIST_DIR


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# ---------------------------------------------------------
# Olist store
# ---------------------------------------------------------

olist_schema_store = Chroma(
    collection_name="olist_schema",
    embedding_function=embeddings,
    persist_directory=OLIST_PERSIST_DIR,
)