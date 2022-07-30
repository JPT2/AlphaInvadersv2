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
agent_memory = agentMemory.Memory()
loss_function = lossFunctions.cross_entropy_with_logits_loss


# Setup configs
training_configs = [
        {'model': models.cnn_model, 'learning_rate': 1e-5, 'episodes': 5000},  # Control
        {'model': models.cnn_model, 'learning_rate': 1e-3, 'episodes': 50},  # Higher learning rate
        # {'model': models.cnn_model, 'learning_rate': 1e-7, 'episodes': 50},  # Lower learning rate
        # {'model': models.cnn_model_smaller, 'learning_rate': 1e-5, 'episodes': 50},  # Simper
        # {'model': models.cnn_model_more_filters, 'learning_rate': 1e-5, 'episodes': 50},  # More filters
        # {'model': models.cnn_model_double_cnn, 'learning_rate': 1e-5, 'episodes': 50},  # Double layer
]
number_of_configs = len(training_configs)

# Track progress
fix, axs = plt.subplots(number_of_configs)


# Define helper functions (later move these to helper / wrapper class)
def predict(model, observation, single=False):
    # TODO add transform to observation to make sure it plays well
    observation = observation if single else np.expand_dims(observation, 0).astype(dtype="float32")
    prediction = model(observation)
    agent_action = tf.random.categorical(prediction, num_samples=1)
    return agent_action.numpy().flatten()[0]


def train(model, optimizer, observations, actions, rewards):
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions, rewards))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(100)

    for batch in dataset.as_numpy_iterator():
        observations = batch[0]
        actions = batch[1]
        rewards = batch[2]

        with tf.GradientTape() as tape:
            predictions = model(np.array(observations))
            losses = loss_function(predictions, np.array(actions), rewards)
        gradients = tape.gradient(losses, model.trainable_variables)

        gradients, _ = tf.clip_by_global_norm(gradients, 0.05)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_model(model, learning_rate, number_of_episodes, view_last_episode=False):
    total_reward_tracker = []
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    while number_of_episodes > 0:
        number_of_episodes -= 1

        if view_last_episode and number_of_episodes % 100 == 0:
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
            action = predict(model, observation)
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
        train(model, optimizer, agent_memory.observations, agent_memory.actions,
              agent_memory.discount_rewards(rewards=agent_memory.rewards))

        # Print out stats
        # if NUM_EPISODES % 5 == 0:
        print("Finished episode. Total rewards was ", agent_memory.total_reward(), ". For an episode of length: ",
              len(agent_memory.rewards), " Episodes remaining: ", number_of_episodes)

    return total_reward_tracker


# Define the training loop
# while number_of_configs > 0:
#     train_model(training_config[0][0], training_config[0][1], training_config[0][2])

while number_of_configs > 0:
    number_of_configs -= 1
    config = training_configs[number_of_configs]
    print("Training with config ", number_of_configs)
    axs[number_of_configs].plot(train_model(config['model'], config['learning_rate'], config['episodes'], True))
plt.show()