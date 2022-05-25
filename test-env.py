import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)

env.close()