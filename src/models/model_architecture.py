import tensorflow as tf
from keras import layers, models, optimizers
from src.utils.model_utils import cosine_similarity

def build_content_based_model(input_dim, output_dim, hidden_layers=2, neurons=64, learning_rate=0.001):
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