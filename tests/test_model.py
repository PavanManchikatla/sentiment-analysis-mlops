import pytest
from src.model.predict import SentimentAnalyzer

@pytest.fixture
def model():
    return SentimentAnalyzer()

def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None

def test_positive_prediction(model):
    """Test prediction on positive text."""
    text = "This is amazing! I love it so much."
    result = model.predict(text)
    
    assert result is not None
    assert "sentiment" in result
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5