import tensorflow as tf
import numpy as np
import os
import json
import argparse
from src.models.model_architecture import build_content_based_model
from src.utils.logging_utils import get_logger
from src.data_processing.data_preprocess import load_data, preprocess_data
from keras.callbacks import EarlyStopping, ModelCheckpoint

logger = get_logger('content_based_algorithm')

class FilteredCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger('model_checkpoint')

    def on_epoch_end(self, epoch, logs=None):
        original_level = self.logger.level
        self.logger.setLevel(logger.ERROR)
        super().on_epoch_end(epoch, logs)
        self.logger.setLevel(original_level)

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        FilteredCallback(
            filepath='best_model',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_format='tf'
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history

def find_similar_tracks(model, track_features, all_features, track_names, n=5):
    track_embedding = model.predict(track_features.reshape(1, -1))
    all_embeddings = model.predict(all_features)
    
    similarities = np.dot(track_embedding, all_embeddings.T)
    top_indices = similarities.argsort()[0][-n:][::-1]
    
    return [(track_names[i], similarities[0][i]) for i in top_indices]

def load_and_preprocess_data(data_path):
    df = load_data(data_path)
    df_processed, _ = preprocess_data(df)
    
    # Assuming the last column is the target variable
    X = df_processed.iloc[:, :-1].values
    y = df_processed.iloc[:, -1].values
    
    return X, y

def main(train_data, val_data, hidden_layers, neurons, embedding_dim, learning_rate, batch_size, dropout_rate):
    try:
        # Load and preprocess data
        X_train, y_train = load_and_preprocess_data(train_data)
        X_val, y_val = load_and_preprocess_data(val_data)
        
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        model = build_content_based_model(
            input_dim, 
            output_dim, 
            hidden_layers=hidden_layers, 
            neurons=neurons, 
            embedding_dim=embedding_dim, 
            learning_rate=learning_rate,
            dropout_rate=dropout_rate
        )
        logger.info("Model built successfully")
        
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=batch_size)
        logger.info("Model training completed")
        
        # Save model metrics
        metrics = {
            "final_loss": history.history['loss'][-1],
            "final_accuracy": history.history['binary_accuracy'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_val_accuracy": history.history['val_binary_accuracy'][-1],
            "val_cosine_similarity": history.history['val_cosine_similarity'][-1]
        }
        
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f)
        logger.info("Model metrics saved")
        
        return model, metrics
    except Exception as e:
        logger.error(f"Error in content-based algorithm main function: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train content-based recommender model')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--hidden_layers', type=int, required=True, help='Number of hidden layers')
    parser.add_argument('--neurons', type=int, required=True, help='Number of neurons per hidden layer')
    parser.add_argument('--embedding_dim', type=int, required=True, help='Dimension of the embedding layer')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--dropout_rate', type=float, required=True, help='Dropout rate for regularization')
    
    args = parser.parse_args()
    
    main(args.train_data, args.val_data, args.hidden_layers, args.neurons, args.embedding_dim, 
        args.learning_rate, args.batch_size, args.dropout_rate)