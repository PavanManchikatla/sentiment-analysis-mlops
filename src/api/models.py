from pydantic import BaseModel, Field
from typing import Dict, Optional

class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    text: str = Field(..., description="Text to analyze")

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    text: str = Field(..., description="Input text")
    sentiment: str = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")