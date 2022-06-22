"""
AlphaDefender - Wrapper for agent

What is the scope for this class?
- Want it to be able to handle the "operations" of the model. Ideally its just a wrapper for all the RL behavior.

So needs to
    - Take an input and give a prediction
    - Initialize / define the model
    - Train itself
    - Log the result / reward of the action for the current state
    - Setup for a new training episode

Expeccted flow
 - Initialize
 - Loop until end of episode
    - Give observation and get result
    - Log reward (Any memory constraints? Do we have to worry about length of episodes?)
 - Train
 - Setup for next episode (Combine with the training step?)


 Evaluation

 - What I got right
    - General methods

- What I missed
    - Preprocessing (or any need for it). Can technically be a substep of predict.
"""


class AlphaDefender:

    def __init__(self):
        # TODO(ptaggs) Initialize the model
        pass

    def predict(self):
        # TODO(ptaggs) Should see if need to define this or if it should be automatic from keras?
        #    Should I inherit from keras?
        pass

    def record_state(self):
        # TODO(ptaggs)
        pass

    def train(self):
        # TODO(ptaggs) Is this something I define explicitly or something tensorflow will handle?
        pass

    def __reset_state__(self):
        # TODO(ptaggs) Clear the memory cache and setup for next episode
        pass


'''
Defines the memory of an agent.

Tracks episodic rewards. Reward for any timestep equal to the reward from current timestep + discount of future rewards.

Evaluation
 - What I got right
    - General structure and high level functions
 
 - What I missed
    - Need to track obervations, ACTIONS, and rewards (Need to remind myself why during impl)
'''


class MemoryCache:

    def __init__(self, discount_factor=0.7):
        # TODO(ptaggs)
        pass

    def record_state(self):
        # TODO(ptaggs)
        pass

    def reset(self):
        # TODO(ptaggs)
        pass

    def get_state(self):
        # TODO(ptaggs)
        pass

    def get_rewards(self):
        # TODO(ptaggs)
        pass