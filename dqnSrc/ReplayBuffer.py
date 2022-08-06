from random import randint, sample


class ReplayBuffer:
    
    def __init__(self, size_of_buffer):
        self.size_of_buffer = size_of_buffer
        self.buffer = []

    def store_transition(self, state, action, reward, next_state):
        if len(self.buffer) > self.size_of_buffer:
            self.buffer.pop(randint(1, self.size_of_buffer))

        self.buffer.append((state, action, reward, next_state))

    def sample(self, n):
        return sample(self.buffer, n)
