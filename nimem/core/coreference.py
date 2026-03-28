import logging
from functools import lru_cache

from returns.result import Result, safe

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_fastcoref_model():
    from fastcoref import FCoref

    logger.info("Loading FastCoref model")
    return FCoref(device="cpu")


@safe
def resolve_coreferences(text: str) -> str:
    model = get_fastcoref_model()
    preds = model.predict(texts=[text])
    return preds[0].get_resolved_text()
