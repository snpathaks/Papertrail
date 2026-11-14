import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from llm import embedder

def cluster_texts(texts, k):
    if len(texts) <= 1:
        return [0] * len(texts)
    k = max(1, min(k, len(texts)))
    try:
        X = np.array(embedder.embed_documents(texts))
    except Exception:
        # Fallback to TF-IDF if embedder is unavailable
        try:
            X = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
            if X.shape[1] == 0:
                X = np.zeros((len(texts), 1))
        except Exception:
            return [i % k for i in range(len(texts))]
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X).tolist()
