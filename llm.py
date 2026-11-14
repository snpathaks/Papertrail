from langchain_ollama import ChatOllama, OllamaEmbeddings
import config
import socket

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

llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL,
                 temperature=0, num_ctx=config.NUM_CTX)
embedder = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)

def ask_structured(schema, prompt, retries=2):
    """Call the LLM with a Pydantic schema; retry on bad output."""
    if not is_ollama_available():
        return None
    runnable = llm.with_structured_output(schema)
    for _ in range(retries + 1):
        try:
            out = runnable.invoke(prompt)
            if out is not None:
                return out
        except Exception:
            pass
    return None

def ask_text(prompt, retries=1):
    if not is_ollama_available():
        return ""
    for _ in range(retries + 1):
        try:
            res = llm.invoke(prompt)
            if hasattr(res, "content"):
                return res.content.strip()
            return str(res).strip()
        except Exception:
            pass
    return ""
