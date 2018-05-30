import random
import numpy as np

from deep_q_network.meta_of_deep_q import float_type, num_units, use_epsilon,\
  epsilon
import tensorflow as tf


class DeepQ():
  
  def __init__(self, sess):
    self.sess = sess
    self.s_batch = tf.placeholder(float_type, [None, num_units])
    self.a_batch = tf.placeholder(float_type, [None, num_units])
    q_input = tf.concat([self.s_batch, self.a_batch], axis=1)
    with tf.variable_scope(name_or_scope="yyx_q_network", reuse=tf.AUTO_REUSE, dtype=float_type):
      w1 = tf.get_variable("fusion1", shape=[2*num_units, num_units])
      w2 = tf.get_variable("fusion2", shape=[num_units, 1])
    q_val_batch_im = tf.matmul(q_input, w1)
    self.q_val_batch = tf.matmul(q_val_batch_im, w2)
    
  def perform_policy(self, s, actions):
    rand_value = random()
    action = None
    if use_epsilon and rand_value < epsilon:
      action = self.random_sample(actions)
    else:
      ss = self.copy_self_and_fill_to_actions(s)
      q_vals = self.compute_q(ss, actions)
      arg_max = np.argmax(q_vals)
      action = actions[arg_max]
    return action
  
  def __call__(self, s_batch, a_batch):
    q_val_batch = self.sess.run([self.q_val_batch], feed_dict = {self.s_batch:s_batch, self.a_batch:a_batch})
    return q_val_batch
