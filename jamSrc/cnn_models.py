import tensorflow as tf
from tensorflow import keras
from keras import layers

# Define a CNN model to accept input from the space invader environment
# Expect pixel data from a screen. Don't expect sprites to be larger than 20 x 20

cnn_model = keras.Sequential([
            layers.Conv2D(filters=5, kernel_size=(20, 20), activation='relu'),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(6, activation=None)
        ])

cnn_model_smaller = keras.Sequential([
            layers.Conv2D(filters=5, kernel_size=(15, 15), activation='relu'),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(6, activation=None)
        ])

cnn_model_more_filters = keras.Sequential([
            layers.Conv2D(filters=7, kernel_size=(20, 20), activation='relu'),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(6, activation=None)
        ])

cnn_model_double_cnn = keras.Sequential([
            layers.Conv2D(filters=10, kernel_size=(20, 20), activation='relu'),
            layers.Conv2D(filters=10, kernel_size=(5, 5), activation='relu'),
            layers.Flatten(),
            layers.Dense(30, activation="relu"),
            layers.Dense(6, activation=None)
        ])