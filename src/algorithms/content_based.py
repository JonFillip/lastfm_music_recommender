import tensorflow as tf
import keras
import logging
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import scikeras
from keras import layers, models, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import uniform, randint
import joblib
import json
from src.utils.model_utils import cosine_similarity


print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

try:
    print(f"Scikeras version: {scikeras.__version__}")
except ImportError:
    print("Scikeras is not installed")


def build_model(input_dim, output_dim, hidden_layers=2, neurons=64, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons, activation='relu'))
    
    model.add(layers.Dense(output_dim, activation='sigmoid', name='embedding'))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', cosine_similarity]
    )
    return model


class FilteredCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a logger
        self.logger = logging.getLogger('tensorflow')
        # Set the logger level to ERROR to suppress INFO messages
        self.logger.setLevel(logging.ERROR)

    def on_epoch_end(self, epoch, logs=None):
        # Temporarily change the logger level
        original_level = self.logger.level
        self.logger.setLevel(logging.ERROR)
        
        # Call the parent method
        super().on_epoch_end(epoch, logs)
        
        # Restore the original logger level
        self.logger.setLevel(original_level)

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32): # change epochs back to 100 after testing
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


def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_dim, output_dim):
    param_distributions = {
        'hidden_layers': randint(1, 4),
        'neurons': randint(32, 129),
        'learning_rate': uniform(0.0001, 0.01),
        'batch_size': randint(16, 65)
    }

    best_val_loss = float('inf')
    best_params = None
    best_model = None

    n_iter = 1  # Number of parameter settings that are sampled 12

    for _ in range(n_iter):
        params = {
            'hidden_layers': param_distributions['hidden_layers'].rvs(),
            'neurons': param_distributions['neurons'].rvs(),
            'learning_rate': param_distributions['learning_rate'].rvs(),
            'batch_size': param_distributions['batch_size'].rvs()
        }
        
        model = build_model(
            input_dim=input_dim, 
            output_dim=output_dim,
            hidden_layers=params['hidden_layers'],
            neurons=params['neurons'],
            learning_rate=params['learning_rate']
        )
        
        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=20,  # Reduced epochs for faster tuning
            batch_size=params['batch_size']
        )
        
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = model

        print(f"Params: {params}, Val Loss: {val_loss}")

    print("Best parameters found: ", best_params)

    # Save the best parameters
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
        
    return best_params, best_model


def find_similar_tracks(model, track_features, all_features, track_names, n=5):
    track_embedding = model.predict(track_features.reshape(1, -1))
    all_embeddings = model.predict(all_features)
    
    similarities = np.dot(track_embedding, all_embeddings.T)
    top_indices = similarities.argsort()[0][-n:][::-1]
    
    return [(track_names[i], similarities[0][i]) for i in top_indices]