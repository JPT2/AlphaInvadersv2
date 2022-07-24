import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import agentMemory
import cnn_models as models
import lossFunctions

# Create and setup the environment
env_hidden = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)
env_human_viewable = gym.make('ALE/SpaceInvaders-v5', full_action_space=False, render_mode='human')

# Initialize any hyper params
NUM_EPISODES = 100
LEARNING_RATE = 1e-5

# Intiqlize other variables
model = models.cnn_model
agent_memory = agentMemory.Memory()
loss_function = lossFunctions.cross_entropy_with_logits_loss
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# Track progress
total_reward_tracker = []

# Define helper functions (later move these to helper / wrapper class)
def predict(observation, single=False):
    # TODO add transform to observation to make sure it plays well
    observation = observation if single else np.expand_dims(observation, 0).astype(dtype="float32")
    prediction = model(observation)
    agent_action = tf.random.categorical(prediction, num_samples=1)
    return agent_action.numpy().flatten()[0]


def train():
    with tf.GradientTape() as tape:
        predictions = model(np.array(agent_memory.observations))
        losses = loss_function(predictions, np.array(agent_memory.actions),
                               agent_memory.discount_rewards(rewards=agent_memory.rewards))
        losses = losses + np.ones(losses.shape)
    gradients = tape.gradient(losses, model.trainable_variables)

    gradients, _ = tf.clip_by_global_norm(gradients, 0.05)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Define the training loop
while NUM_EPISODES > 0:
    NUM_EPISODES -= 1

    if NUM_EPISODES % 100 == 0:
        env = env_human_viewable
    else:
        env = env_hidden
    # Reset the environment, reset training variables
    agent_memory.clear()
    observation = env.reset()
    done = False
    penalty_timer = 15

    observation_to_store = observation
    while not done:
        action = predict(observation)
        observation, reward, done, info = env.step(action)

        # Add penalty for dragging on for too long (should it be based on if the model's scored a reward recently? Is that getting too specific?)
        if reward > 0:
            penalty_timer = 15
        else:
            penalty_timer -= 1
        if penalty_timer == 0:
            reward -= 10
        # Write rewards to memory buffer so we can use them to get reward for a particular timestep
        agent_memory.add_to_memory(new_observation=observation_to_store, new_action=action, new_reward=reward)
        observation_to_store = observation  # Want to store the obseravtion that generates the action

    # Add a large negative penalty for the agent "losing"
    agent_memory.rewards[-1] -= 500

    # Log total reaward to plot
    total_reward_tracker.append(agent_memory.total_reward())
    train()

    # Print out stats
    # if NUM_EPISODES % 5 == 0:
    print("Finished episode. Total rewards was ", agent_memory.total_reward(), ". Episodes remaining: ", NUM_EPISODES)

plt.plot(total_reward_tracker)
plt.show()