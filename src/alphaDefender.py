"""
AlphaDefender - Wrapper for agent

What is the scope for this class?
- Want it to be able to handle the "operations" of the model. Ideally its just a wrapper for all the RL behavior.

So needs to
    - Take an input and give a prediction
    - Initialize / define the model
    - Train itself
    - Log the result / reward of the action for the current state
    - Setup for a new training episode

Expeccted flow
 - Initialize
 - Loop until end of episode
    - Give observation and get result
    - Log reward (Any memory constraints? Do we have to worry about length of episodes?)
 - Train
 - Setup for next episode (Combine with the training step?)


 Evaluation

 - What I got right
    - General methods

- What I missed
    - Preprocessing (or any need for it). Can technically be a substep of predict.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5

class AlphaDefender:

    def __init__(self):
        self.model = keras.Sequential([
            # Define CNN layer with 20x15 filter (roughly size of most objects)
            # Define 5 filters (detect self, bullets, enemies, barrier, and another for mutations)
            layers.Conv2D(filters=5, kernel_size=(20, 15), activation='relu'),
            layers.Flatten(),
            layers.Dense(100, activation="relu"),
            layers.Dense(4, activation=None)
        ])

        self.actions = [[NOOP, RIGHT, LEFT], [FIRE, RIGHTFIRE, LEFTFIRE]]

    def predict(self, observation, single=True):
        # TODO(ptaggs) Should see if need to define this or if it should be automatic from keras?
        #    Should I inherit from keras?
        observation = np.expand_dims(observation, axis=0) if single else observation
        model_predictions = self.model(observation)
        actions = tf.map_fn(self.__get_action, model_predictions, dtype=tf.int32)
        return actions[0] if single else actions

    def __get_action(self, predictions):
        should_shoot = tf.sigmoid(predictions[0]) >= np.random.random()
        movement_action = tf.random.categorical([predictions[1:3]], num_samples=1)[0]
        return self.actions[int(should_shoot)][movement_action[0]]

    def train(self, loss_function, optimizer, observations, actions, discounted_rewards):
        with tf.GradientTape() as tape:
            predictions = self.model(observations) # Going to generate predictions based on all the steps
            loss = loss_function(predictions, actions, discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)

        # Need to clip the gradients to avoid falling into local minima?
        grads, _ = tf.clip_by_global_norm(grads, 0.5) # TODO Twiddle with this hyper param
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
