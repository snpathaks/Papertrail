import chromadb
import config
from llm import embedder

_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
_col = _client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

def paper_indexed(uid):
    return len(_col.get(where={"paper_id": uid}, limit=1)["ids"]) > 0

def add_chunks(uid, chunks):
    if not chunks or paper_indexed(uid):
        return
    texts = [c["text"] for c in chunks]
    _col.add(
        ids=[f"{uid}::{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embedder.embed_documents(texts),
        metadatas=[{"paper_id": uid, "section": c["section"], "page": c["page"]} for c in chunks],
    )

def retrieve(uid, query, k=3):
    res = _col.query(query_embeddings=[embedder.embed_query(query)],
                     n_results=k, where={"paper_id": uid})
    return [{"text": d, **m} for d, m in zip(res["documents"][0], res["metadatas"][0])]