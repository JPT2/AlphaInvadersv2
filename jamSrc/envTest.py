# Test file for playing with the environment

import gym

WILL_RENDER = True

# Program definition
if WILL_RENDER:
    env = gym.make('ALE/SpaceInvaders-v5', full_action_space=False, render_mode='human')
else:
    env = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)

observation = env.reset()
done = False

print(env.action_space)

while not done:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)