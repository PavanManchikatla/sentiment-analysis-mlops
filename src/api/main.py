from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from prometheus_client import Counter, Histogram, start_http_server
import uvicorn
import os
from src.api.models import SentimentRequest, SentimentResponse
from src.model.predict import SentimentAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create metrics
REQUESTS = Counter("sentiment_api_requests_total", "Total number of requests")
PREDICTIONS = Counter("sentiment_predictions_total", "Total number of predictions", ["sentiment"])
PREDICTION_TIME = Histogram("sentiment_prediction_time_seconds", "Time spent processing prediction")

# Start Prometheus metrics server on a separate port
start_http_server(9090)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using a pretrained model from Hugging Face",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model singleton
model = None

def get_model():
    """Singleton pattern to load model only once."""
    global model
    if model is None:
        model = SentimentAnalyzer()
    return model

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    get_model()
    logger.info("API started - model loaded")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest, model: SentimentAnalyzer = Depends(get_model)):
    """
    Predict sentiment from text input.
    
    Args:
        request: Request body containing the text for analysis
        
    Returns:
        dict: Prediction results
    """
    REQUESTS.inc()
    
    try:
        with PREDICTION_TIME.time():
            result = model.predict(request.text)
        
        # Update metrics
        PREDICTIONS.labels(sentiment=result["sentiment"]).inc()
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)