import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LLM Provider switch: "ollama" (default) | "groq" | "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Model settings
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Vector store backend: "memory" (NumPy) | "chroma" (ChromaDB)
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "memory").lower()
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# OpenAlex settings
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")

# Operational limits
NUM_CTX = 4096
MAX_PAPERS = 10
MIN_PAPERS = 5
MAX_SEARCH_ATTEMPTS = 2
PDF_DOWNLOAD_TIMEOUT = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "30"))  # 30s timeout
PDF_MAX_SIZE_MB = int(os.getenv("PDF_MAX_SIZE_MB", "15"))          # 15 MB cap
PDF_MAX_SIZE_BYTES = PDF_MAX_SIZE_MB * 1024 * 1024

# Storage paths
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
RUNS_DIR = Path("runs")
for d in (DATA_DIR, PDF_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "papertrail.db"
CHROMA_DIR = str(DATA_DIR / "chroma")
