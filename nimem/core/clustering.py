from typing import List, Dict
from returns.result import Result, safe
import numpy as np
import logging

try:
    logging.info("Attempting to import fast_hdbscan...")
    from fast_hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None

@safe
def perform_clustering(vectors: np.ndarray, texts: List[str], min_cluster_size: int = 2) -> Dict[int, List[str]]:
    """
    Clusters pre-computed embedding vectors and maps them back to their text labels.
    Embedding is the caller's responsibility.
    """
    if not texts:
        return {}
        
    if HDBSCAN is None:
        raise ImportError("fast_hdbscan is not installed.")

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(vectors)
    
    clusters: Dict[int, List[str]] = {}
    for text, label in zip(texts, labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    return clusters

def generate_topic_name(texts: List[str]) -> str:
    """
    Simple heuristic to name the cluster.
    """
    return "Topic: " + ", ".join(list(set(texts))[:3])
