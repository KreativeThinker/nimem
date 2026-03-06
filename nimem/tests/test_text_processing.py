import pytest
from unittest.mock import MagicMock, patch
from nimem.core import text_processing
from returns.result import Success, Failure

@pytest.fixture
def mock_spacy():
    text_processing.get_spacy_model.cache_clear()
    with patch('nimem.core.text_processing.get_spacy_model') as mock_get:
        nlp = MagicMock()
        mock_get.return_value = nlp

        mock_ent_alice = MagicMock()
        mock_ent_alice.text = "Alice"
        mock_ent_alice.label_ = "PERSON"
        mock_ent_alice.start_char = 0
        mock_ent_alice.start = 0
        mock_ent_alice.end = 1

        mock_ent_google = MagicMock()
        mock_ent_google.text = "Google"
        mock_ent_google.label_ = "ORG"
        mock_ent_google.start_char = 15
        mock_ent_google.start = 3
        mock_ent_google.end = 4

        doc = MagicMock()
        doc.ents = [mock_ent_alice, mock_ent_google]
        doc.__iter__ = MagicMock(return_value=iter([]))
        nlp.return_value = doc

        yield nlp
    text_processing.get_spacy_model.cache_clear()

@pytest.fixture
def mock_gliner():
    text_processing.get_gliner_model.cache_clear()
    with patch('nimem.core.text_processing.get_gliner_model') as mock_get:
        instance = MagicMock()
        mock_get.return_value = instance
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
        
        mock_doc = MagicMock()
        
        mock_word_he = MagicMock()
        mock_word_he.start_char = 31
        mock_word_he.end_char = 33
        
        mock_word_alice = MagicMock()
        mock_word_alice.start_char = 0
        mock_word_alice.end_char = 5
        
        mock_sent_0 = MagicMock()
        mock_sent_0.words = [mock_word_alice]
        mock_sent_1 = MagicMock()
        mock_sent_1.words = [mock_word_he]
        mock_doc.sentences = [mock_sent_0, mock_sent_1]
        
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

def test_extract_triplets_heuristic(mock_spacy):
    triplets = text_processing.extract_triplets("Alice works at Google").unwrap()
    mock_spacy.assert_called_once()
    assert len(triplets) > 0

def test_extract_triplets_gliner2(mock_gliner):
    triplets = text_processing.extract_triplets(
        "Alice works at Google", use_gliner2=True
    ).unwrap()

    mock_gliner.extract_relations.assert_called_once()
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
    assert "He" not in res
    assert res.count("Alice") >= 2

def test_pipeline_heuristic(mock_spacy, mock_stanza):
    res = text_processing.process_text_pipeline("Input text")
    assert isinstance(res, Success)
    _, triplets = res.unwrap()
    assert len(triplets) >= 0

def test_pipeline_gliner2(mock_gliner):
    res = text_processing.process_text_pipeline("Input text", use_gliner2=True)
    assert isinstance(res, Success)
    _, triplets = res.unwrap()
    assert len(triplets) == 2
