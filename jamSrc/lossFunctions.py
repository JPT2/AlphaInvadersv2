import tensorflow as tf

def cross_entropy_with_logits_loss(logits, actions, discounted_rewards):
 neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
 return tf.reduce_mean(neg_logprob * discounted_rewards, 0)