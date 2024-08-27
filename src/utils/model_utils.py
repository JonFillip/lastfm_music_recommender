import tensorflow as tf
import keras
import logging
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import json
from keras import layers, models, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from src.algorithms.content_based import find_similar_tracks


# Utility functions

def prepare_data(df_svd, original_df):
    X = df_svd.values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(original_df['similar_tracks'].str.split(','))
    track_names = original_df['name'].values
    
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, track_names, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, names_train, names_test, scaler, mlb


def cosine_similarity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)

def save_model_artifacts(model, scaler, mlb, feature_names, base_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(base_path, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model in the new .keras format
    model_path = os.path.join(model_dir, "saved_model")
    model.save(model_path, save_format='tf')
    print(f"Model saved to {model_path}")

    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(mlb, os.path.join(model_dir, "mlb.joblib"))
    
    with open(os.path.join(model_dir, "model_summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    with open(os.path.join(model_dir, "feature_names.txt"), 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Model and artifacts saved to {model_dir}")
    return model_dir

def load_model_artifacts(model_dir):
    full_model_dir = os.path.join(model_dir, 'saved_model')
    
    print(f"Attempting to load artifacts from: {full_model_dir}")

    # Load model
    model_path = full_model_dir
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found in {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={'cosine_similarity': cosine_similarity})
    print(f"Model loaded from {model_path}")

    # Load scaler
    scaler_path = os.path.join(full_model_dir, "..", "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    # Load MultiLabelBinarizer
    mlb_path = os.path.join(full_model_dir, "..", "mlb.joblib")
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"MLB file not found: {mlb_path}")
    mlb = joblib.load(mlb_path)
    print(f"MultiLabelBinarizer loaded from {mlb_path}")

    # Load feature names
    feature_names_path = os.path.join(full_model_dir, "..", "feature_names.txt")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    print(f"Feature names loaded from {feature_names_path}")

    return model, scaler, mlb, feature_names

def visualize_training_history(history):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['cosine_similarity'])
    plt.plot(history.history['val_cosine_similarity'])
    plt.title('Cosine Similarity')
    plt.ylabel('Similarity')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def test_content_based_model(model_dir, X_test, names_test, y_test):
    try:
        loaded_model, loaded_scaler, loaded_mlb, loaded_feature_names = load_model_artifacts(model_dir)
        print("All artifacts loaded successfully.")

        test_loss, test_accuracy, test_cosine_similarity = loaded_model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Cosine Similarity: {test_cosine_similarity}")

        # Find similar tracks example
        example_track_index = 0
        similar_tracks = find_similar_tracks(loaded_model, X_test[example_track_index], X_test, names_test)
        print(f"Similar tracks to '{names_test[example_track_index]}':")
        for name, similarity in similar_tracks:
            print(f"{name}: {similarity}")

    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")