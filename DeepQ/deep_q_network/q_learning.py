import tensorflow as tf
from deep_q_network.meta_of_deep_q import float_type, num_units, gamma
from deep_q_network.deep_q import DeepQ
from deep_q_network.replay_memory import D


class QLearn():
  
  def __init__(self, sess):
    self.sess = sess
    self.action_value = DeepQ(sess)
    self.target_action_value = DeepQ(sess)
    
    self.s_t_batch = tf.placeholder(float_type, [None, num_units])
    self.a_t_batch = tf.placeholder(float_type, [None, num_units])
    self.r_t_batch = tf.placeholder(float_type, [None])
    self.s_t_1_batch = tf.placeholder(float_type, [None, num_units])
    self.s_t_1_actions_batch = tf.placeholder(float_type, [None, None, num_units])
    
    action_batch = self.target_action_value.perform_policy(self.s_t_1_batch, self.s_t_1_actions_batch)
    y_batch = self.r_t_batch + gamma * self.target_action_value(self.s_t_1_actions_batch, action_batch)

    action_value_q_vals = self.action_value(self.s_t_batch, self.a_t_batch)
    
    self.loss = tf.losses.mean_squared_error(y_batch, action_value_q_vals)
    self.train = self.adam.minimize(self.loss)
    
  def __call__(self, s_t_batch, a_t_batch, r_t_batch, s_t_1_batch, s_t_1_actions_batch):
    loss_val, _ = self.sess.run([self.loss, self.train], feed_dict={self.s_t_batch:s_t_batch, self.a_t_batch:a_t_batch, self.r_t_batch:r_t_batch, self.s_t_1_batch:s_t_1_batch, self.s_t_1_actions_batch:s_t_1_actions_batch})
    return loss_val

def randoop_interact():
  pass

def __main__():
  with tf.Session() as sess:
    M = 10
    T = 10
    d = D()
    for _ in range(M):
      for _ in range(T):
        s_t, a_t, r_t, s_t_1 = randoop_interact()
        d.store_transition(s_t, a_t, r_t, s_t_1)
        this_turn_loss_value = QLearn(sess)(d.sample_minibatch())
        print(this_turn_loss_value)
