import chromadb
import config
from llm import embedder

_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
_col = _client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

def paper_indexed(uid):
    try:
        return len(_col.get(where={"paper_id": uid}, limit=1)["ids"]) > 0
    except Exception:
        return False

def add_chunks(uid, chunks):
    if not chunks or paper_indexed(uid):
        return
    texts = [c["text"] for c in chunks]
    metadatas = [{"paper_id": uid, "section": c["section"], "page": c["page"]} for c in chunks]
    ids = [f"{uid}::{i}" for i in range(len(chunks))]
    try:
        embeddings = embedder.embed_documents(texts)
        _col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    except Exception:
        try:
            _col.add(ids=ids, documents=texts, metadatas=metadatas)
        except Exception:
            pass

def retrieve(uid, query, k=3):
    try:
        all_ids = _col.get(where={"paper_id": uid})["ids"]
        count = len(all_ids)
        if count == 0:
            return []
        k_val = min(k, count)
        try:
            em = embedder.embed_query(query)
            res = _col.query(query_embeddings=[em], n_results=k_val, where={"paper_id": uid})
        except Exception:
            try:
                res = _col.query(query_texts=[query], n_results=k_val, where={"paper_id": uid})
            except Exception:
                records = _col.get(where={"paper_id": uid}, limit=k_val)
                return [{"text": d, **m} for d, m in zip(records["documents"], records["metadatas"])]

        if res and res.get("documents") and len(res["documents"]) > 0:
            return [{"text": d, **m} for d, m in zip(res["documents"][0], res["metadatas"][0])]
        return []
    except Exception:
        return []
