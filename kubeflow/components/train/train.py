from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
)
from src.algorithms.content_based import main as train_content_based
from src.utils.logging_utils import get_logger

logger = get_logger('kubeflow_train')

@component(
    packages_to_install=['tensorflow', 'numpy', 'pandas', 'scikit-learn'],
    base_image='python:3.9'
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_hyperparameters: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics]
) -> float:
    import json
    import pandas as pd
    
    try:
        # Load hyperparameters
        with open(best_hyperparameters.path, 'r') as f:
            hyperparams = json.load(f)
        
        # Load data
        train_df = pd.read_csv(train_data.path)
        val_df = pd.read_csv(val_data.path)
        
        # Train model
        trained_model, model_metrics = train_content_based(
            train_df,
            val_df,
            hidden_layers=int(hyperparams['hidden_layers']),
            neurons=int(hyperparams['neurons']),
            embedding_dim=int(hyperparams['embedding_dim']),
            learning_rate=float(hyperparams['learning_rate']),
            batch_size=int(hyperparams['batch_size']),
            dropout_rate=float(hyperparams['dropout_rate'])
        )
        
        # Save model
        trained_model.save(model.path)
        logger.info(f"Model saved to {model.path}")
        
        # Save metrics
        with open(metrics.path, 'w') as f:
            json.dump(model_metrics, f)
        logger.info(f"Metrics saved to {metrics.path}")
        
        return model_metrics['val_cosine_similarity']
    
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train model component for Kubeflow')
    parser.add_argument('--train_data', type=str, help='Path to training dataset')
    parser.add_argument('--val_data', type=str, help='Path to validation dataset')
    parser.add_argument('--best_hyperparameters', type=str, help='Path to best hyperparameters')
    parser.add_argument('--model', type=str, help='Path to save the trained model')
    parser.add_argument('--metrics', type=str, help='Path to save the model metrics')
    
    args = parser.parse_args()
    
    train_model(
        train_data=args.train_data,
        val_data=args.val_data,
        best_hyperparameters=args.best_hyperparameters,
        model=args.model,
        metrics=args.metrics
    )