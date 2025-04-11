# Sentiment Analysis MLOps Project

An end-to-end MLOps pipeline demonstrating how to deploy and monitor a machine learning model in production. This project uses a pretrained sentiment analysis model from Hugging Face and implements the complete MLOps lifecycle.

## Features

- 🤗 **Pretrained Model**: Uses Hugging Face's BERT-based sentiment analysis model
- 🚀 **FastAPI**: High-performance API for model serving
- 📊 **Monitoring**: Prometheus metrics for tracking model performance and system health
- 🔄 **CI/CD**: Automated testing with GitHub Actions
- 🐳 **Docker**: Containerization for consistent deployment
- 📁 **DVC**: Data version control for tracking datasets

## Project Structure

```
sentiment-analysis-mlops/
├── .github/
│   └── workflows/         # CI/CD workflows
│       └── ci.yaml        # GitHub Actions configuration
├── data/
│   ├── raw/               # Raw datasets (tracked by DVC)
│   └── processed/         # Processed datasets
├── models/
│   └── model_registry/    # Model storage
├── src/
│   ├── api/
│   │   ├── main.py        # FastAPI application
│   │   └── models.py      # Pydantic request/response models
│   ├── model/
│   │   ├── config.py      # Model configuration
│   │   └── predict.py     # Prediction logic
│   ├── monitoring/
│   │   └── metrics.py     # Prometheus metrics
│   └── utils/
│       └── logger.py      # Logging utilities
├── tests/
│   └── test_model.py      # Model tests
├── .dvc/                  # DVC configuration
├── .gitignore             # Git ignore file
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker and Docker Compose (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PavanManchikatla/sentiment-analysis-mlops.git
   cd sentiment-analysis-mlops
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Pull the sample data with DVC:
   ```bash
   dvc pull
   ```
   Note: If you don't have a DVC remote configured, you can recreate the sample data:
   ```bash
   python create_sample_data.py
   ```

### Running the API Locally

Start the FastAPI application:
```bash
uvicorn src.api.main:app --reload
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs

### Running with Docker

1. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

2. The API will be available at http://localhost:8000
3. Prometheus metrics are exposed at http://localhost:9090

## API Endpoints

- `POST /predict`: Analyze the sentiment of a text
  ```json
  {
    "text": "This product is amazing, I love it!"
  }
  ```

- `GET /health`: Health check endpoint

## Testing

Run the tests with pytest:
```bash
pytest
```

## Monitoring

This project uses Prometheus for monitoring. Key metrics include:
- `sentiment_api_requests_total`: Total number of API requests
- `sentiment_predictions_total`: Prediction counts by sentiment
- `sentiment_prediction_time_seconds`: Prediction latency

## CI/CD Pipeline

The GitHub Actions workflow in `.github/workflows/ci.yaml` automatically:
1. Runs tests on every push to the main branch
2. Checks code quality
3. Verifies the build process

## Future Enhancements

- Model drift detection with Evidently AI
- A/B testing capability
- Automatic model retraining
- Grafana dashboards for visualization
- Kubernetes deployment configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
