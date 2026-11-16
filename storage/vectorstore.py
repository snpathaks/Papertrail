import numpy as np
import config
from llm import embedder

class MemoryVectorStore:
    """In-memory NumPy vector store for zero-dependency deployment (e.g. Streamlit Cloud)."""
    def __init__(self):
        # uid -> list of dicts: {"text", "section", "page", "paper_id", "vector"}
        self._store = {}

    def paper_indexed(self, uid):
        return uid in self._store and len(self._store[uid]) > 0

    def add_chunks(self, uid, chunks):
        if not chunks or self.paper_indexed(uid):
            return
        texts = [c["text"] for c in chunks]
        try:
            vectors = embedder.embed_documents(texts)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            try:
                vecs = TfidfVectorizer(max_features=384).fit_transform(texts).toarray()
                vectors = [v.tolist() for v in vecs]
            except Exception:
                vectors = [[0.0] * 384 for _ in texts]

        stored = []
        for c, v in zip(chunks, vectors):
            vec = np.array(v, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            stored.append({
                "text": c["text"],
                "section": c.get("section", "abstract"),
                "page": c.get("page", 1),
                "paper_id": uid,
                "vector": vec,
            })
        self._store[uid] = stored

    def retrieve(self, uid, query, k=3):
        chunks = self._store.get(uid, [])
        if not chunks:
            return []
        k_val = min(k, len(chunks))
        try:
            q_raw = embedder.embed_query(query)
            q_vec = np.array(q_raw, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            # Cosine similarity using normalized dot product
            scores = [float(np.dot(q_vec, c["vector"])) for c in chunks]
            top_idx = np.argsort(scores)[::-1][:k_val]
            return [{k: c[k] for k in ("text", "section", "page", "paper_id")} for c in [chunks[i] for i in top_idx]]
        except Exception:
            # Fallback to TF-IDF cosine similarity if embedder query fails
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                all_texts = [c["text"] for c in chunks] + [query]
                tfidf = TfidfVectorizer().fit_transform(all_texts).toarray()
                doc_vecs = tfidf[:-1]
                q_vec = tfidf[-1]
                scores = [float(np.dot(q_vec, d)) for d in doc_vecs]
                top_idx = np.argsort(scores)[::-1][:k_val]
                return [{k: c[k] for k in ("text", "section", "page", "paper_id")} for c in [chunks[i] for i in top_idx]]
            except Exception:
                return [{k: c[k] for k in ("text", "section", "page", "paper_id")} for c in chunks[:k_val]]


class ChromaVectorStore:
    """Persistent ChromaDB vector store."""
    def __init__(self):
        import chromadb
        self._client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self._col = self._client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

    def paper_indexed(self, uid):
        try:
            return len(self._col.get(where={"paper_id": uid}, limit=1)["ids"]) > 0
        except Exception:
            return False

    def add_chunks(self, uid, chunks):
        if not chunks or self.paper_indexed(uid):
            return
        texts = [c["text"] for c in chunks]
        metadatas = [{"paper_id": uid, "section": c.get("section", ""), "page": c.get("page", 1)} for c in chunks]
        ids = [f"{uid}::{i}" for i in range(len(chunks))]
        try:
            embeddings = embedder.embed_documents(texts)
            self._col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        except Exception:
            try:
                self._col.add(ids=ids, documents=texts, metadatas=metadatas)
            except Exception:
                pass

    def retrieve(self, uid, query, k=3):
        try:
            all_ids = self._col.get(where={"paper_id": uid})["ids"]
            count = len(all_ids)
            if count == 0:
                return []
            k_val = min(k, count)
            try:
                em = embedder.embed_query(query)
                res = self._col.query(query_embeddings=[em], n_results=k_val, where={"paper_id": uid})
            except Exception:
                try:
                    res = self._col.query(query_texts=[query], n_results=k_val, where={"paper_id": uid})
                except Exception:
                    records = self._col.get(where={"paper_id": uid}, limit=k_val)
                    return [{"text": d, **m} for d, m in zip(records["documents"], records["metadatas"])]

            if res and res.get("documents") and len(res["documents"]) > 0:
                return [{"text": d, **m} for d, m in zip(res["documents"][0], res["metadatas"][0])]
            return []
        except Exception:
            return []


# Initialize selected backend
_backend_choice = getattr(config, "VECTOR_BACKEND", "memory").lower()
if _backend_choice == "chroma":
    try:
        _store = ChromaVectorStore()
    except Exception as e:
        print(f"[Warning] Failed to initialize ChromaDB ({e}). Falling back to in-memory NumPy store.")
        _store = MemoryVectorStore()
else:
    _store = MemoryVectorStore()

def paper_indexed(uid):
    return _store.paper_indexed(uid)

def add_chunks(uid, chunks):
    return _store.add_chunks(uid, chunks)

def retrieve(uid, query, k=3):
    return _store.retrieve(uid, query, k=k)
