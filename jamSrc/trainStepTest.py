import numpy as np
import tensorflow as tf

import lossFunctions
import cnn_models
import gym

env = gym.make('ALE/SpaceInvaders-v5', full_action_space=False)

observation = env.reset()
observation = np.expand_dims(observation, axis=0).astype(dtype="float32")
prediction = cnn_models.cnn_model(observation)
# Have the model predict log probabilities for each action. Sample from the distribution defined
action = tf.random.categorical(prediction, num_samples=1)
action = action.numpy().flatten()
print("Action: ", action)
_, reward, done, info = env.step(action[0])
print("Reward is: ", reward)
loss = lossFunctions.cross_entropy_with_logits_loss(prediction, action, reward)
print("Loss is ", loss)

# Define a train step
LEARNING_RATE = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
with tf.GradientTape() as tape:
    predictions = cnn_models.cnn_model(observation)  # Going to generate predictions based on all the steps
    loss = lossFunctions.cross_entropy_with_logits_loss(predictions, action, reward)

print("Pre train")
print(cnn_models.cnn_model.trainable_variables)
grads = tape.gradient(loss, cnn_models.cnn_model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, 0.5)
optimizer.apply_gradients(zip(grads, cnn_models.cnn_model.trainable_variables))
print("Post train")
print(cnn_models.cnn_model.trainable_variables)