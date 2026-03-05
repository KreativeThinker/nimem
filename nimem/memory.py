from returns.result import Result, Success, Failure
import logging

logging.info("Importing nimem.core.text_processing...")
from .core import text_processing
logging.info("Importing nimem.core.embeddings...")
from .core import embeddings
logging.info("Importing nimem.core.graph_store...")
from .core import graph_store
logging.info("Importing nimem.core.clustering...")
from .core import clustering
logging.info("Importing nimem.core.schema...")
from .core import schema
from typing import Dict, List

def ingest_text(text: str) -> Result[str, Exception]:
    processed: Result = text_processing.process_text_pipeline(text)
    def store_triplets(data) -> Result[str, Exception]:
        resolved_text, triplets = data
        logging.info(f"Resolved Text: {resolved_text}")
        logging.info(f"Found Triplets: {len(triplets)}")
        
        count = 0
        errors = []
        for tri in triplets:
             logging.info(f"Adding: {tri.subject} -[{tri.relation}]-> {tri.object}")
             
             cardinality = schema.CARDINALITY.get(tri.relation, "MANY")
             if cardinality == "ONE":
                 logging.info(f"Relation '{tri.relation}' is 1-to-1. Expiring old facts.")
                 res_expire = graph_store.expire_facts(tri.subject, tri.relation)
                 if isinstance(res_expire, Failure):
                     logging.warning(f"Failed to expire facts for {tri.subject}/{tri.relation}: {res_expire}")
             
             res = graph_store.add_fact(tri.subject, tri.relation, tri.object)
             if isinstance(res, Success):
                 count += 1
             else:
                 errors.append(str(res.failure()))
        
        if errors and count == 0:
            return Failure(RuntimeError(f"All {len(errors)} facts failed to store: {errors}"))
        
        if errors:
            logging.warning(f"Some facts failed to store: {errors}")
            
        return Success(f"Ingested {count} facts. (Resolved text: {resolved_text[:50]}...)")

    return processed.bind(store_triplets)

def add_memory(subject: str, relation: str, obj: str) -> Result[bool, Exception]:
    """
    Directly adds a memory fact.
    """
    return graph_store.add_fact(subject, relation, obj)

def recall_memory(subject: str) -> Result[list, Exception]:
    """
    Recalls facts about a subject.
    """
    return graph_store.query_valid_facts(subject)

def consolidate_topics() -> Result[str, Exception]:
    """
    Performs clustering on all entity names in the graph to find 'weak' relations (Topics).
    Creates 'BELONGS_TO' edges from Entities to new Topic nodes.
    """
    entities_res = graph_store.get_all_entities()
    
    def embed_and_cluster(entities: List[str]):
        return embeddings.embed_texts(entities).bind(
            lambda vectors: clustering.perform_clustering(vectors, entities)
        )
    
    return entities_res.bind(embed_and_cluster).map(_process_clusters)

def _process_clusters(clusters: Dict[int, List[str]]) -> str:
    count = 0
    topic_count = 0
    for label, items in clusters.items():
        if label == -1: continue
        topic_name = clustering.generate_topic_name(items)
        logging.info(f"Found Cluster '{topic_name}': {items}")
        topic_count += 1
        
        for item in items:
            res = graph_store.add_fact(item, "BELONGS_TO", topic_name)
            if isinstance(res, Success):
                count += 1
            else:
                logging.warning(f"Failed to add cluster relation for {item}: {res}")
            
    return f"Consolidated {count} weak relations into {topic_count} topics."
