import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = tf.ones((1, 210, 160, 3))

# Define the model
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()

# Model A - single layer
# Input layer is a CNN
# Second layer is 100 neurons
# Output layer is 4 neurons
model_a = keras.Sequential([
    layers.Conv2D(filters=4, kernel_size=(20, 10)),
    layers.Flatten(),
    layers.Dense(100,  activation="relu"),
    layers.Dense(4,  activation=None)
])
print(model_a(x))
print(model_a.summary())

# Model B - stacked layers
# Input layer is a CNN (4 x (20x10))
# Second layer is 50 neruons
# Third layer is 50 neurons
# Output layer is 4 neurons (Shoot, Left, NOOP, Right)
model_b = keras.Sequential([
    layers.Conv2D(filters=4, kernel_size=(20, 10)),
    layers.Flatten(),
    layers.Dense(50, activation="relu"),
    layers.Dense(50, activation="relu"),
    layers.Dense(4, activation=None)
])
print(model_b(x))
print(model_b.summary())

class AlphaInvader(keras.layers):
    def __init__(self):
        # TODO define the model here (maybe take a flag? or wrap this to override)
        pass

    # TODO Convert raw model output to env understandable action
    def _get_action(self):
        pass