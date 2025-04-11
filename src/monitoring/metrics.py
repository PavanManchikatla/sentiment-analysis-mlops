from prometheus_client import Counter, Histogram, Gauge
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Request metrics
REQUEST_COUNT = Counter(
    "api_requests_total", 
    "Total number of API requests",
    ["endpoint", "method", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Model metrics
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of model predictions",
    ["sentiment"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "model_confidence_histogram",
    "Distribution of model confidence scores",
    ["sentiment"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)

def record_request(endpoint, method, status, latency):
    """Record API request metrics."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    
def record_prediction(sentiment, confidence):
    """Record model prediction metrics."""
    PREDICTION_COUNT.labels(sentiment=sentiment).inc()
    CONFIDENCE_HISTOGRAM.labels(sentiment=sentiment).observe(confidence)