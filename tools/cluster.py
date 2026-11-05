import numpy as np
from sklearn.cluster import KMeans
from llm import embedder

def cluster_texts(texts, k):
    if len(texts) <= 1:
        return [0] * len(texts)
    k = max(1, min(k, len(texts)))
    X = np.array(embedder.embed_documents(texts))
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X).tolist()