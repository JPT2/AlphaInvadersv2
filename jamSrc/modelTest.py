import gym
import cnn_models
import numpy as np

import gym


env = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)

observation = env.reset()
print(np.shape(observation))
# Need to add another dimension before can feed to model
# How do I avoid so many array copies?
observation = np.expand_dims(observation, axis=0).astype(dtype="float32")
print(np.shape(observation))
prediction = cnn_models.cnn_model(observation)
# Prediction comes out as 6 values (floats, can be positive or negative)
print(prediction)
print(np.shape(prediction))