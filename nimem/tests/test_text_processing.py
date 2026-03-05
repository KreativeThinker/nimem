import pytest
from unittest.mock import MagicMock, patch
from nimem.core import text_processing
from returns.result import Success, Failure

@pytest.fixture
def mock_gliner():
    text_processing.get_gliner_model.cache_clear()
    with patch('nimem.core.text_processing.GLiNER2') as mock_cls:
        instance = MagicMock()
        mock_cls.from_pretrained.return_value = instance
        instance.extract_relations.return_value = {
            'relation_extraction': {
                'knows': [('Alice', 'Bob')],
                'works_for': [{'head': {'text': 'Alice'}, 'tail': {'text': 'Google'}}]
            }
        }
        yield instance
    text_processing.get_gliner_model.cache_clear()

@pytest.fixture
def mock_fcoref():
    text_processing.get_fastcoref_model.cache_clear()
    with patch('nimem.core.text_processing.FCoref') as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        preds = MagicMock()
        preds.get_resolved_text.return_value = "Alice works at Google. Alice knows Bob."
        instance.predict.return_value = [preds]
        yield instance
    text_processing.get_fastcoref_model.cache_clear()

def test_extract_triplets_success(mock_gliner):
    triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
    assert len(triplets) == 2
    
    assert triplets[0].subject == 'Alice'
    assert triplets[0].relation == 'knows'
    assert triplets[0].object == 'Bob'
    
    assert triplets[1].subject == 'Alice'
    assert triplets[1].relation == 'works_for'
    assert triplets[1].object == 'Google'

def test_resolve_coreferences(mock_fcoref):
    res = text_processing.resolve_coreferences("She works at Google.").unwrap()
    assert res == "Alice works at Google. Alice knows Bob."

def test_pipeline_integration(mock_gliner, mock_fcoref):
    res = text_processing.process_text_pipeline("Input text")
    assert isinstance(res, Success)
    resolved, triplets = res.unwrap()
    assert resolved == "Alice works at Google. Alice knows Bob."
    assert len(triplets) == 2
