from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import logging
from src.utils.logger import setup_logger
from src.model.config import DEFAULT_MODEL_NAME

logger = setup_logger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        """Initialize the model with a pretrained model from Hugging Face."""
        try:
            # Check if there's a fine-tuned model in the model registry
            self.model_path = os.getenv("MODEL_PATH", None)
            
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                logger.info(f"Loading pretrained model {model_name} from Hugging Face")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Map of sentiment labels
            self.id2label = {
                0: "negative",
                1: "positive"
            }
            
            logger.info(f"Model loaded successfully. Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, text):
        """
        Perform sentiment analysis on the input text.
        
        Args:
            text (str): Input text for sentiment analysis
            
        Returns:
            dict: Prediction results with sentiment and confidence score
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get prediction
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Log inference for monitoring
            logger.info(f"Inference: '{text[:50]}...' -> {self.id2label[predicted_class]} ({confidence:.4f})")
            
            return {
                "text": text,
                "sentiment": self.id2label[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    self.id2label[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise