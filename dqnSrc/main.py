import gym
import numpy as np
import tensorflow as tf

import Dqn
import ReplayBuffer

# Declare hyper params
EPISODES = 10
MIN_EPISODE_LENGTH = 10

REPLAY_BUFFER_SIZE = 1000
MINI_BATCH_SIZE = 100

LEARNING_RATE = 1e-3

CLIP = 0.05

# Initializatopms
replay_buffer = ReplayBuffer.ReplayBuffer(REPLAY_BUFFER_SIZE)
env_hidden = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)
model = Dqn.Dqn(len(env_hidden.action_space))
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# Helper functions
def preprocess(observation):
    return observation

def loss_function(states, actions, rewards, next_states):
    # Use Bellman function
    # L() = (y - Q(s,a))^2
    # y = r + max_{a'}Q(s',a')
    predictions = model.predict(states)
    q = predictions[actions]
    predictions_prime = model.predict(next_states)
    y = predictions_prime[np.argmax(predictions_prime)] + rewards
    return (y - q)^2

def train_model(optimizer, model, replay_buffer):
    mini_batch = replay_buffer.sample(MINI_BATCH_SIZE)
    with tf.GradientTape() as tape:
        actions = model.get_action(mini_batch[:,0], off_policy=False)
        losses = loss_function(actions, state=mini_batch[:,0], next_state=mini_batch[:,1], rewards=mini_batch[:,2])
    gradients = tape.gradient(losses, model.trainable_variables)
    gradients = tf.clip_by_global_norm(gradients, CLIP)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Train the model
# Train for specified number of episodes
while EPISODES > 0:
    # Reset the environment
    observation = preprocess(env.reset())
    done = False
    current_episode_length = 0
    # Start episode (guarantee minimum length?)
    while not done and current_episode_length < MIN_EPISODE_LENGTH:
        # Select our action (epsilon greedy) - DQN.GetAction (K steps?)
        action = model.get_action(observation)
        # Step forward through environment
        next_observation, reward, done, info = env.step(action)
        # Get new state, preprocess
        next_observation = preprocess(next_observation)
        # Store transition in replay buffer
        replay_buffer.store_transition(observation, action, reward, next_observation)
        # Perform training step
        train_model(model, replay_buffer)
        current_episode_length += 1
    EPISODES -= 1 