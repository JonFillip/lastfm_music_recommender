import argparse
import os
from src.algorithms.content_based import ContentBasedRecommender
from src.utils.logging_utils import setup_logging
from src.utils.model_utils import save_model

logger = setup_logging()

def train_model(data_path, model_path, hyperparameters):
    logger.info("Starting model training")
    
    # Load data
    # Assume data is loaded and preprocessed here
    
    # Initialize and train the model
    model = ContentBasedRecommender(**hyperparameters)
    model.fit(data)
    
    # Save the model
    save_model(model, model_path)
    
    logger.info(f"Model training completed. Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the music recommender model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--hyperparameters', type=str, required=True, help='JSON string of hyperparameters')
    
    args = parser.parse_args()
    
    # Convert hyperparameters from JSON string to dictionary
    import json
    hyperparameters = json.loads(args.hyperparameters)
    
    train_model(args.data_path, args.model_path, hyperparameters)