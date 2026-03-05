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
def mock_stanza():
    text_processing.get_coref_pipeline.cache_clear()
    with patch('nimem.core.text_processing.stanza') as mock_mod:
        mock_pipe = MagicMock()
        mock_mod.Pipeline.return_value = mock_pipe
        
        # Build a mock document with coref chains
        mock_doc = MagicMock()
        
        # Mock a word for "He" mention (non-representative)
        mock_word_he = MagicMock()
        mock_word_he.start_char = 31
        mock_word_he.end_char = 33
        
        # Mock words for "Alice" mention (representative)
        mock_word_alice = MagicMock()
        mock_word_alice.start_char = 0
        mock_word_alice.end_char = 5
        
        # Mock sentence containing both
        mock_sent_0 = MagicMock()
        mock_sent_0.words = [mock_word_alice]
        mock_sent_1 = MagicMock()
        mock_sent_1.words = [mock_word_he]
        mock_doc.sentences = [mock_sent_0, mock_sent_1]
        
        # Mock CorefMention and CorefChain
        mention_rep = MagicMock()
        mention_rep.sentence = 0
        mention_rep.start_word = 0
        mention_rep.end_word = 0
        
        mention_he = MagicMock()
        mention_he.sentence = 1
        mention_he.start_word = 0
        mention_he.end_word = 0
        
        mock_chain = MagicMock()
        mock_chain.representative_text = "Alice"
        mock_chain.representative_index = 0
        mock_chain.mentions = [mention_rep, mention_he]
        
        mock_doc.coref = [mock_chain]
        mock_pipe.return_value = mock_doc
        
        yield mock_pipe
    text_processing.get_coref_pipeline.cache_clear()

def test_extract_triplets_success(mock_gliner):
    triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
    assert len(triplets) == 2
    
    assert triplets[0].subject == 'Alice'
    assert triplets[0].relation == 'knows'
    assert triplets[0].object == 'Bob'
    
    assert triplets[1].subject == 'Alice'
    assert triplets[1].relation == 'works_for'
    assert triplets[1].object == 'Google'

def test_resolve_coreferences(mock_stanza):
    text = "Alice works at Google. Alice knows Bob. He is happy."
    res = text_processing.resolve_coreferences(text).unwrap()
    # "He" (chars 31-33) should be replaced with "Alice"
    assert "He" not in res
    assert res.count("Alice") >= 2

def test_pipeline_integration(mock_gliner, mock_stanza):
    res = text_processing.process_text_pipeline("Input text")
    assert isinstance(res, Success)
    resolved, triplets = res.unwrap()
    assert len(triplets) == 2
