import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers

class Dqn:

    def __init__(self, output_size):
        self.model = tf.keras.Sequential([
            layers.Conv2D(filters=16, kernel_size=(8, 8), activation='relu'),
            layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu'),
            layers.Dense(256, activaion='relu'),
            layers.Dense(output_size)
        ])
        self.epsilon = 1
        self.action_space_size = output_size

    def get_action(self, observation, off_policy=True):
        if off_policy:
            if self.epsilon > random.random():
                return random.randint(0, self.output_size - 1)
        predictions = self.model(observation)
        return np.argmax(predictions)