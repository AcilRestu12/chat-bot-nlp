import tensorflow as tf
# from tensorflow.keras import layers, models
from keras import layers, models

def create_model(input_size, hidden_size, output_size):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_size,)))
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.Dense(output_size, activation='softmax'))
    return model

