import cv2
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human', repeat_action_probability=0)
observation, info = env.reset(seed=42, return_info=True)

print("Observation spcae: ", env.observation_space)
print("Observation shape: ", np.shape(observation))
print("Action space: ", env.action_space)
print("Reward ranges: ", env.reward_range)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)

env.close()