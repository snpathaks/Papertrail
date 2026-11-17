import os
import time
import socket
import config

def is_ollama_available():
    try:
        url_part = config.OLLAMA_URL.split("://")[-1]
        host = url_part.split(":")[0]
        port = int(url_part.split(":")[1].split("/")[0])
        s = socket.socket()
        s.settimeout(0.5)
        res = s.connect_ex((host, port))
        s.close()
        return res == 0
    except Exception:
        return False

def is_rate_limit(e):
    if hasattr(e, "status_code") and e.status_code == 429:
        return True
    if hasattr(e, "response") and getattr(e.response, "status_code", None) == 429:
        return True
    err_str = str(e).lower()
    return any(m in err_str for m in ("429", "rate limit", "resource_exhausted", "too many requests"))

# 1. LLM Provider Setup
provider = config.LLM_PROVIDER.lower()

if provider == "groq":
    from langchain_groq import ChatGroq
    api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY") or "gsk_placeholder"
    llm = ChatGroq(model=config.GROQ_MODEL, api_key=api_key, temperature=0)
elif provider == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "placeholder"
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=api_key, temperature=0)
else:  # default "ollama"
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL,
                     temperature=0, num_ctx=config.NUM_CTX)

# 2. Embeddings Provider Setup
class FastEmbedWrapper:
    """Wrapper around fastembed TextEmbedding to match langchain Embeddings interface."""
    def __init__(self, model_name=config.FASTEMBED_MODEL):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        if not texts:
            return []
        try:
            return [e.tolist() for e in self._model.embed(texts)]
        except Exception:
            return []

    def embed_query(self, query):
        try:
            res = list(self._model.embed([query]))
            return res[0].tolist() if res else []
        except Exception:
            return []

if provider == "ollama":
    from langchain_ollama import OllamaEmbeddings
    embedder = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)
else:
    embedder = FastEmbedWrapper(model_name=config.FASTEMBED_MODEL)

# 3. Structured Output & Text Helpers with Rate Limit Handling
def ask_structured(schema, prompt, retries=3, base_delay=2.0):
    """Call the LLM with a Pydantic schema; retries on 429 rate limits or bad output."""
    if provider == "ollama" and not is_ollama_available():
        return None

    try:
        runnable = llm.with_structured_output(schema)
    except Exception:
        return None

    for attempt in range(retries + 1):
        try:
            out = runnable.invoke(prompt)
            if out is not None:
                return out
        except Exception as e:
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
                continue
            break
    return None

def ask_text(prompt, retries=3, base_delay=2.0):
    """Call the LLM for plain text output; retries on 429 rate limits or transient errors."""
    if provider == "ollama" and not is_ollama_available():
        return ""

    for attempt in range(retries + 1):
        try:
            res = llm.invoke(prompt)
            if hasattr(res, "content"):
                return res.content.strip()
            return str(res).strip()
        except Exception as e:
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
                continue
            break
    return ""
