from numpy.random import random

import src.alphaDefender as AlphaDefenderSrc
import gym


'''
Goals:
    [Done] - Get the environment to run with random action
    [Done] - Get the environment to run using random actions from the model
    - Get the environment to use the model to get the actions
    - Add training step
    - Have model run an episode and train after
'''

# Get the model to train
model = AlphaDefenderSrc.AlphaDefender()
env = gym.make('ALE/SpaceInvaders-v5', render_mode='human', full_action_space=False)

# Setup training loop

# Define hyperparams
EPISODES = 1 # 00  # How many episodes to run to train the model
MIN_EPISODE_LENGTH = 100 # How many steps we need to make in the smallest possible episode

for i in range(EPISODES):
    # Initialize the environment
    observation = env.reset() # TODO(ptaggs) Figure out if need to set a seed
    done = False
    episode_length = 0

    # Start the episode (Do we need a max episode length?)
    while not done or episode_length < MIN_EPISODE_LENGTH:
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)

    # Train the model
    model.train()