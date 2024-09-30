from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    Artifact
)
from typing import NamedTuple
from src.algorithms.content_based import main as train_content_based
from src.utils.logging_utils import setup_logger, log_error, log_step

logger = setup_logger('kubeflow_train')

OutputSpec = NamedTuple('Outputs', [
    ('val_cosine_similarity', float),
    ('val_ndcg', float),
    ('model_version', str)
])

@component(
    packages_to_install=['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'],
    base_image='python:3.10'
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    best_hyperparameters: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    training_plots: Output[Artifact]
) -> OutputSpec:
    import json
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    
    try:
        log_step(logger, "Loading hyperparameters", "Model Training")
        with open(best_hyperparameters.path, 'r') as f:
            hyperparams = json.load(f)
        
        log_step(logger, "Loading data", "Model Training")
        train_df = pd.read_csv(train_data.path)
        val_df = pd.read_csv(val_data.path)
        
        log_step(logger, "Setting up callbacks", "Model Training")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_cosine_similarity', 
            patience=5, 
            mode='max', 
            restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model.path,
            monitor='val_cosine_similarity',
            mode='max',
            save_best_only=True
        )
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )
        
        log_step(logger, "Training model", "Model Training")
        trained_model, model_metrics, history = train_content_based(
            train_df,
            val_df,
            hidden_layers=int(hyperparams['hidden_layers']),
            neurons=int(hyperparams['neurons']),
            embedding_dim=int(hyperparams['embedding_dim']),
            learning_rate=float(hyperparams['learning_rate']),
            batch_size=int(hyperparams['batch_size']),
            dropout_rate=float(hyperparams['dropout_rate']),
            callbacks=[early_stopping, model_checkpoint, lr_scheduler]
        )
        
        log_step(logger, "Saving model", "Model Training")
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        trained_model.save(f"{model.path}_{model_version}")
        logger.info(f"Model saved to {model.path}_{model_version}")
        
        log_step(logger, "Saving metrics", "Model Training")
        with open(metrics.path, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics.path}")
        
        log_step(logger, "Creating training plots", "Model Training")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training history
        ax1.plot(history.history['cosine_similarity'], label='Train Cosine Similarity')
        ax1.plot(history.history['val_cosine_similarity'], label='Val Cosine Similarity')
        ax1.set_title('Model Cosine Similarity')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Plot learning rate
        ax2.plot(history.history['lr'], label='Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.set_ylabel('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_yscale('log')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(training_plots.path)
        logger.info(f"Training plots saved to {training_plots.path}")
        
        return OutputSpec(
            val_cosine_similarity=model_metrics['val_cosine_similarity'],
            val_ndcg=model_metrics['val_ndcg'],
            model_version=model_version
        )
    
    except Exception as e:
        log_error(logger, e, 'Model Training')
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train model component for Kubeflow')
    parser.add_argument('--train_data', type=str, help='Path to training dataset')
    parser.add_argument('--val_data', type=str, help='Path to validation dataset')
    parser.add_argument('--best_hyperparameters', type=str, help='Path to best hyperparameters')
    parser.add_argument('--model', type=str, help='Path to save the trained model')
    parser.add_argument('--metrics', type=str, help='Path to save the model metrics')
    parser.add_argument('--training_plots', type=str, help='Path to save the training plots')
    
    args = parser.parse_args()
    
    train_model(
        train_data=args.train_data,
        val_data=args.val_data,
        best_hyperparameters=args.best_hyperparameters,
        model=args.model,
        metrics=args.metrics,
        training_plots=args.training_plots
    )