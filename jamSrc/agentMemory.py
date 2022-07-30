import numpy as np
import tensorflow as tf

def create_batched_dataset_from_episode(observations, actions, rewards):
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions, rewards))
    dataset = dataset.shuffle(buffer_size=100)
    return dataset.batch(100)

class Memory:
    def __init__(self):
        self.full_clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Wipes short and longer term memory
    def full_clear(self):
        self.best_episodes = []
        self.clear()

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    # Naively store out best episodes so far (and potentially worst episodes)
    # Stores up to 5 episodes, after that, lowest score episode will be dropped
    def update_best_episodes(self, observations, actions, rewards, total_rewards):
        if len(self.best_episodes) < 5:
            # Convert to dataset
            self.best_episodes.append({
                'total_rewards': total_rewards,
                'dataset': create_batched_dataset_from_episode(observations, actions, rewards)})
            if len(self.best_episodes) == 5:
                self.best_episodes.sort(key = lambda x : x['total_rewards'])
            return

        # Should use a min heap?
        if total_rewards < self.best_episodes[0]['total_rewards']:
            return

        index_to_insert = 5
        for episode in reversed(self.best_episodes):
            if episode['total_rewards'] < total_rewards:
                self.best_episodes.insert(index_to_insert, {
                    'total_rewards': total_rewards,
                    'dataset': create_batched_dataset_from_episode(observations, actions, rewards)})
                self.best_episodes.pop(0)
                return
            index_to_insert -=1

    def __len__(self):
        return len(self.actions)

    def __normalize(self, x):
        x -= np.mean(x)
        x /= np.abs(np.std(x))
        return x.astype(np.float32)

    # Compute normalized, discounted, cumulative rewards (i.e., return)
    # Arguments:
    #   rewards: reward at timesteps in episode
    #   gamma: discounting factor
    # Returns:
    #   normalized discounted reward
    def discount_rewards(self, rewards, gamma=0.95):
        discounted_rewards = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(0, len(rewards))):
            # update the total discounted reward
            R = R * gamma + rewards[t]
            discounted_rewards[t] = R

        return self.__normalize(discounted_rewards)

    def total_reward(self):
        cum_reward = 0
        for reward in self.rewards:
            cum_reward += reward
        return cum_reward