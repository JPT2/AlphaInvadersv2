from numpy.random import random

import gym
import tensorflow as tf
import numpy as np

import src.alphaDefender as AlphaDefenderSrc
import src.memory as MemorySrc

'''
Goals:
    [Done] - Get the environment to run with random action
    [Done] - Get the environment to run using random actions from the model
    [Done] - Get the environment to use the model to get the actions
    - Add training step
    - Have model run an episode and train after
'''

def loss_function(logits, actions, discounted_rewards):
    # Loss is a function of the actions we took and the rewards we got for them
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    product = neg_logprob * discounted_rewards
    loss = tf.reduce_mean(neg_logprob * discounted_rewards)
    return -10 if loss == 0 else loss

# Get the model to train
model = AlphaDefenderSrc.AlphaDefender()
# env = gym.make('ALE/SpaceInvaders-v5', render_mode='human', full_action_space=False)
env = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)

LEARNING_RATE = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Setup training loop

# Define hyperparams
EPISODES = 1000 # 00  # How many episodes to run to train the model
MIN_EPISODE_LENGTH = 100 # How many steps we need to make in the smallest possible episode

memory_cache = MemorySrc.Memory()

for i in range(EPISODES):
    # Initialize the environment
    observation = env.reset() # TODO(ptaggs) Figure out if need to set a seed
    done = False
    episode_length = 0

    # Start the episode (Do we need a max episode length?)
    while episode_length < 1: # not done or episode_length < MIN_EPISODE_LENGTH:
        action = model.predict(tf.cast(observation, tf.float32))
        observation, reward, done, info = env.step(action)
        if reward != 0:
            print("Non-zero reward! ", reward)
        memory_cache.add_to_memory(new_observation=observation, new_action=action, new_reward=reward)

        if done:
            # Do some updating to model memory and reset the env
            observation = env.reset()
        episode_length += 1

    # Train the model
    model.train(loss_function, optimizer, np.array(memory_cache.observations),
                np.array(memory_cache.actions), memory_cache.discount_rewards(memory_cache.rewards))
    if EPISODES % 2 == 0:
        print("Finished an epoch! Reward for last episode: ", memory_cache.total_reward())
    memory_cache.clear()