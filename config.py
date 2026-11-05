import os
from pathlib import Path

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")  # for OpenAlex

NUM_CTX = 4096
MAX_PAPERS = 10
MIN_PAPERS = 5
MAX_SEARCH_ATTEMPTS = 2

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
for d in (DATA_DIR, PDF_DIR):
    d.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "papertrail.db"
CHROMA_DIR = str(DATA_DIR / "chroma")