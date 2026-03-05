from typing import List, Tuple, NamedTuple
from functools import lru_cache
from returns.result import Result, Success, Failure, safe
import logging

try:
    logging.info("Attempting to import gliner2...")
    from gliner2 import GLiNER2
except ImportError:
    logging.warning("gliner2 not found. Please ensure it is installed.")
    GLiNER2 = None

try:
    logging.info("Attempting to import stanza...")
    import stanza
except ImportError:
    logging.warning("stanza not found. Please ensure it is installed.")
    stanza = None

class Triple(NamedTuple):
    subject: str
    relation: str
    object: str

from . import schema

@lru_cache(maxsize=1)
def get_gliner_model():
    if GLiNER2 is None:
        raise ImportError("gliner2 library not installed.")
    logging.info("Loading GLiNER2 model: urchade/gliner_small-v2.1 ...")
    return GLiNER2.from_pretrained("urchade/gliner2-multi-v1")

@lru_cache(maxsize=1)
def get_coref_pipeline():
    if stanza is None:
        raise ImportError("stanza library not installed.")
    logging.info("Loading Stanza coref pipeline (en)...")
    return stanza.Pipeline("en", processors="tokenize,coref")

@safe
def extract_triplets(text: str) -> List[Triple]:
    """
    Extracts triplets using GLiNER2 relation extraction.
    """
    model = get_gliner_model()
    
    relation_types = list(schema.RELATIONS.keys())
    
    try:
        results = model.extract_relations(text, relation_types)
    except AttributeError as e:
        raise RuntimeError(f"GLiNER model does not support extract_relations: {e}") from e
    
    triplets = []
    relations_map = results.get('relation_extraction', {})
    
    for rel_type, instances in relations_map.items():
        for instance in instances:
            if isinstance(instance, tuple) or isinstance(instance, list):
                if len(instance) >= 2:
                    triplets.append(Triple(instance[0], rel_type, instance[1]))
            elif isinstance(instance, dict):
                 head = instance.get('head', {}).get('text', '')
                 tail = instance.get('tail', {}).get('text', '')
                 if head and tail:
                     triplets.append(Triple(head, rel_type, tail))

    return triplets

@safe
def resolve_coreferences(text: str) -> str:
    """
    Resolves coreferences using Stanza's coref pipeline.
    Replaces non-representative mentions with their representative text.
    """
    pipe = get_coref_pipeline()
    doc = pipe(text)
    
    if not doc.coref:
        return text
    
    replacements = []
    for chain in doc.coref:
        rep_text = chain.representative_text
        for mention in chain.mentions:
            if mention == chain.mentions[chain.representative_index]:
                continue
            
            sent = doc.sentences[mention.sentence]
            words = sent.words[mention.start_word:mention.end_word + 1]
            start_char = words[0].start_char
            end_char = words[-1].end_char
            replacements.append((start_char, end_char, rep_text))
    
    replacements.sort(key=lambda x: x[0], reverse=True)
    resolved = text
    for start, end, rep in replacements:
        resolved = resolved[:start] + rep + resolved[end:]
    
    return resolved

def process_text_pipeline(text: str) -> Result[Tuple[str, List[Triple]], Exception]:
    """
    Chains coreference resolution -> Triplet Extraction.
    Uses bind to propagate errors safely without unwrap.
    """
    return resolve_coreferences(text).bind(
        lambda resolved: extract_triplets(resolved).map(
             lambda triplets: (resolved, triplets)
        )
    )
