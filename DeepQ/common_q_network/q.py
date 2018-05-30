from random import random


epsilon = 0.8
gamma=0.9
alpha=0.1
use_epsilon = True

class CommonQ():
  
  def __init__(self):
    self.Q = {}
  
  def perform_policy(self, state, action_space):
    rand_value = random()
    action = None
    if use_epsilon and rand_value < epsilon:  
      action = self.random_sample(action_space)
    else:
      action = self.compute_max_action(state, action_space)
    return action
  
  '''
  act should be done in java
  def act(self, a):
    return self.env.step(a)
  '''
  
  def learning_one_action(self, s0, a0, s1, a1, r1):
    old_q = self.get_Q_value(s0, a0)
    q_prime = self.get_Q_value(s1, a1)
    td_target = r1 + gamma * q_prime  
    new_q = old_q + alpha * (td_target - old_q)
    self.set_Q_value(s0, a0, new_q)

    
  def get_Q_value(self, s, a):
    
    pass
  
  def set_Q_value(self, s, a, v):
    
    pass
    
  def random_sample(self, action_space):
    
    pass
    
  def compute_max_action(self, state, action_space):
    '''
    deep fitting network
    '''
    
    pass
    