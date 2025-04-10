FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model_registry/latest
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "src/api/main.py"]