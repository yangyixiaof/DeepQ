import numpy as np


minibatch_size = 10

class D():
  
  def __init__(self):
    self.transitions = []
  
  def store_transition(self, s_t, a_t, r_t, s_t_1):
    self.transitions.append((s_t, a_t, r_t, s_t_1))

  def sample_minibatch(self):
    transition_size = len(self.transitions)
    norm_size = min(minibatch_size, transition_size)
    minibatch = np.random.permutation(transition_size)[0:norm_size]
    return minibatch
