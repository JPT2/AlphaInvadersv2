import numpy as np

class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    # Add observations, actions, rewards to memory
    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)

    def __len__(self):
        return len(self.actions)

    def __normalize(self, x):
        x -= np.mean(x)
        x /= np.std(x)
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
