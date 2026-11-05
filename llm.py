from langchain_ollama import ChatOllama, OllamaEmbeddings
import config

llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL,
                 temperature=0, num_ctx=config.NUM_CTX)
embedder = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)

def ask_structured(schema, prompt, retries=2):
    """Call the LLM with a Pydantic schema; retry on bad output."""
    runnable = llm.with_structured_output(schema)
    for _ in range(retries + 1):
        try:
            out = runnable.invoke(prompt)
            if out is not None:
                return out
        except Exception:
            pass
    return None

def ask_text(prompt):
    return llm.invoke(prompt).content.strip()