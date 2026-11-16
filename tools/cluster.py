import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from llm import embedder

def cluster_texts(texts, k):
    """Cluster a list of texts into k clusters, with automatic TF-IDF fallback."""
    if not texts:
        return []
    if len(texts) <= 1:
        return [0] * len(texts)
    k = max(1, min(k, len(texts)))

    X = None
    try:
        vecs = embedder.embed_documents(texts)
        if vecs and len(vecs) == len(texts):
            X = np.array(vecs, dtype=np.float32)
    except Exception:
        X = None

    # Fallback automatically to TF-IDF if embeddings fail or produce invalid vectors
    if X is None or X.size == 0 or np.all(X == 0):
        try:
            tfidf = TfidfVectorizer(stop_words="english")
            X = tfidf.fit_transform(texts).toarray()
            if X.shape[1] == 0:
                X = np.zeros((len(texts), 1))
        except Exception:
            return [i % k for i in range(len(texts))]

    try:
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X).tolist()
    except Exception:
        return [i % k for i in range(len(texts))]
