import tensorflow as tf
import numpy as np
import os
import json
from src.models.model_architecture import build_content_based_model
from src.utils.logging_utils import get_logger
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

def main(X_train, y_train, X_val, y_val, input_dim, output_dim):
    try:
        model = build_content_based_model(input_dim, output_dim)
        logger.info("Model built successfully")
        
        history = train_model(model, X_train, y_train, X_val, y_val)
        logger.info("Model training completed")
        
        # Save model metrics
        metrics = {
            "final_loss": history.history['loss'][-1],
            "final_accuracy": history.history['binary_accuracy'][-1],
            "final_val_loss": history.history['val_loss'][-1],
            "final_val_accuracy": history.history['val_binary_accuracy'][-1]
        }
        
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f)
        logger.info("Model metrics saved")
        
        return model
    except Exception as e:
        logger.error(f"Error in content-based algorithm main function: {e}")
        raise

if __name__ == "__main__":
    # This section would be replaced by Kubeflow pipeline component inputs
    # For now, we'll just log a message
    logger.info("Content-based algorithm script executed. This would be replaced by Kubeflow component execution.")