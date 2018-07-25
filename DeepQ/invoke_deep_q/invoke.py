import json
import socket

import numpy as np
import tensorflow as tf

gamma = 0.9
num_units = 128
bool_type = tf.bool
int_type = tf.int32
float_type = tf.float32

action_value_prefix = ""
target_action_value_prefix = "target_"

API_Maximum_Number = 1000
num_units = 128
Branch_Maximum_Number = 1000


def variable_initialize():
  with tf.variable_scope(name_or_scope="yyx_q_network", reuse=tf.AUTO_REUSE, dtype=float_type):
    '''
    parameters to compute embeds
    '''
    tf.get_variable("embeds_var", shape=[API_Maximum_Number, num_units])
    tf.get_variable("embed_w", shape=[num_units, num_units])
    tf.get_variable("embed_action_w", shape=[num_units, num_units])
    tf.get_variable("embed_action_up_down_w", shape=[2 * num_units, num_units])
    '''
    parameters to compute q networks
    '''
    tf.get_variable(action_value_prefix + "fusion1", shape=[Branch_Maximum_Number, 2 * num_units, num_units])
    tf.get_variable(action_value_prefix + "fusion2", shape=[Branch_Maximum_Number, num_units, 1])
    tf.get_variable(target_action_value_prefix + "fusion1", shape=[Branch_Maximum_Number, 2 * num_units, num_units])
    tf.get_variable(target_action_value_prefix + "fusion2", shape=[Branch_Maximum_Number, num_units, 1])
  
  
class EmbedComputer():
  
  def __init__(self):
    self.info = "info"
  
  def compute_states_actions_embed(self, s_batch, s_segment_batch, a_batch, a_segment_batch):
  #   s_batch = tf.placeholder(int_type, [2, None])
  #   s_segment_batch = tf.placeholder(int_type, [None])
  #   a_batch = tf.placeholder(int_type, [2, None])
  #   a_segment_batch = tf.placeholder(int_type, [None])
    
    '''
    compute state actions embed
    '''
    
    def compute_state_actions_cond(i, i_len, *_):
      return tf.less(i, i_len)
          
    def compute_state_actions_body(i, i_len, s_embed_batch, a_embed_batch, a_embed_segment_batch):
      s_range_start = tf.cond(tf.equal(i, tf.constant(0, int_type)), lambda: tf.constant(0, int_type), lambda: s_segment_batch[i - 1])
      s_range_end = s_segment_batch[i]
      one_s = tf.slice(s_batch, [0, s_range_start], [2, (s_range_end - s_range_start)])
      one_s_embed = self.compute_state_embed(one_s)
      s_embed_batch = tf.concat([s_embed_batch, [one_s_embed[-1]]], axis=0)
      
      a_range_start = tf.cond(tf.equal(i, tf.constant(0, int_type)), lambda: tf.constant(0, int_type), lambda: a_segment_batch[i - 1])
      a_range_end = a_segment_batch[i]
      one_s_acts = tf.slice(a_batch, [0, a_range_start], [2, (a_range_end - a_range_start)])
      one_s_acts_embed = self.compute_action_embed(one_s_embed, one_s_acts)
      a_embed_batch = tf.concat([a_embed_batch, one_s_acts_embed], axis=0)
      a_embed_segment_start = tf.cond(tf.equal(i, tf.constant(0, int_type)), lambda: tf.constant(0, int_type), lambda: a_embed_segment_batch[i - 1])
      a_embed_segment_batch = tf.concat([a_embed_segment_batch, [a_embed_segment_start + tf.shape(one_s_acts_embed)[0]]], axis=0)
      return i + 1, i_len, s_embed_batch, a_embed_batch, a_embed_segment_batch
    
    i = tf.constant(0, int_type)
    i_len = tf.shape(s_segment_batch)[-1]
    s_embed_batch = tf.zeros([0, num_units], float_type)
    a_embed_batch = tf.zeros([0, num_units], float_type)
    a_embed_segment_batch = tf.zeros([0], int_type)
    _, _, s_embed_batch, a_embed_batch, a_embed_segment_batch = tf.while_loop(compute_state_actions_cond, compute_state_actions_body, [i, i_len, s_embed_batch, a_embed_batch, a_embed_segment_batch], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, num_units]), tf.TensorShape([None, num_units]), tf.TensorShape([None])])
    return s_embed_batch, a_embed_batch, a_embed_segment_batch
    
  def compute_state_embed(self, compute_tensor):
#     compute_tensor = tf.Print(compute_tensor, [compute_tensor], "compute_tensor")
    
    with tf.variable_scope(name_or_scope="yyx_q_network", reuse=tf.AUTO_REUSE, dtype=float_type):
      '''
      parameters to compute embeds
      '''
      embeds_var = tf.get_variable("embeds_var")
      embed_w = tf.get_variable("embed_w")
#     one_stmt = compute_tensor[i]
    
    def iterate_denpency_cond(d, d_len, *_):
#       return tf.cond(tf.less(d, d_len), lambda: tf.cond(tf.greater_equal(one_stmt[0][d], tf.constant(0, int_type)), lambda: tf.constant(True, bool_type), lambda: tf.constant(False, bool_type)), lambda: tf.constant(False, bool_type))
      return tf.less(d, d_len)
    
    def iterate_denpency_body(d, d_len, one_stmt_embed, output_embed):
#       output_embed = tf.Print(output_embed, [output_embed, tf.shape(embeds_var), compute_tensor], "output_embed/tf.shape(embeds_var)#in_iterate_denpency_body", summarize=100)
      one_embed = tf.cond(tf.equal(compute_tensor[1][d], tf.constant(0, int_type)), lambda: output_embed[compute_tensor[0][d]], lambda: embeds_var[compute_tensor[0][d]])
      one_embed = tf.expand_dims(one_embed, axis=0)
      one_stmt_embed = tf.tanh(tf.add(tf.matmul(one_embed, embed_w), one_stmt_embed))
      d = d + 1
      one_stmt_embed, output_embed = tf.cond(tf.less(d, d_len), lambda: tf.cond(tf.equal(compute_tensor[1][d], tf.constant(2, int_type)), lambda: self.sequence_part_over(one_stmt_embed, output_embed), lambda: (one_stmt_embed, output_embed)), lambda: self.sequence_part_over(one_stmt_embed, output_embed))
      return d, d_len, one_stmt_embed, output_embed
    
    d = tf.constant(0, int_type)
    d_len = tf.shape(compute_tensor)[-1]
    one_stmt_embed = tf.zeros([1, num_units], float_type)
    output_embed = tf.zeros([0, num_units], float_type)
    _, _, one_stmt_embed, output_embed = tf.while_loop(iterate_denpency_cond, iterate_denpency_body, [d, d_len, one_stmt_embed, output_embed], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([None, num_units])], parallel_iterations=1)
#     embeds = embeds.write(i, output_embed)
    return output_embed
#     return i, i_len, embeds
    
#     i = tf.constant(0, int_type)
#     i_len = tf.shape(compute_tensor)[0]
#     embeds = tf.TensorArray(float_type, size=0, dynamic_size=True, clear_after_read=False, element_shape=[1, num_units])
#     _, _, embeds = tf.while_loop(compute_embed_cond, compute_embed_body, [i, i_len, embeds])
#     embeds_tensor = embeds.stack(name="compute_embeds")
#     return embeds_tensor

  def sequence_part_over(self, one_embed, output_embed):
    output_embed = tf.concat([output_embed, one_embed], axis=0)  
    one_embed = tf.zeros([1, num_units], float_type)
    return one_embed, output_embed
    
  def compute_action_embed(self, state_embed, compute_tensor):
  #   compute_tensor = tf.placeholder(int_type, [2, None], "compute_action_tensor")
    total_embed = tf.expand_dims(state_embed[-1], axis=0)
    
    def compute_embed_cond(i, i_len, *_):
      return tf.less(i, i_len)
      
    def compute_embed_body(i, i_len, one_act_embed, output_embed):
      with tf.variable_scope(name_or_scope="yyx_q_network", reuse=tf.AUTO_REUSE, dtype=float_type):
        '''
        parameters to compute embed
        '''
        embeds_var = tf.get_variable("embeds_var")
        embed_action_w = tf.get_variable("embed_action_w")
        embed_action_up_down_w = tf.get_variable("embed_action_up_down_w")
      
      sig = compute_tensor[0][i]
      flg = compute_tensor[1][i]
      
      def handle_statement_index():
        one_up_embed = tf.cond(tf.greater(sig, tf.constant(0, int_type)), lambda: tf.expand_dims(state_embed[sig - 1], axis=0), lambda: tf.zeros([1, num_units], float_type))
        one_down_embed = total_embed - one_up_embed
        to_compute_embed = tf.concat([one_up_embed, one_down_embed], axis=1)
        return tf.matmul(to_compute_embed, embed_action_up_down_w)
      
      def handle_reference_or_elements():
        return tf.cond(tf.equal(flg, tf.constant(0, int_type)), lambda: tf.expand_dims(state_embed[sig], axis=0), lambda: tf.expand_dims(embeds_var[sig], axis=0))
      
      one_embed = tf.cond(tf.equal(flg, tf.constant(2, int_type)), handle_statement_index, handle_reference_or_elements)
      one_act_embed = tf.tanh(tf.add(tf.matmul(one_embed, embed_action_w), one_act_embed))
      
      i = i + 1
      one_act_embed, output_embed = tf.cond(tf.less(i, i_len), lambda: tf.cond(tf.equal(compute_tensor[1][i], tf.constant(2, int_type)), lambda: self.sequence_part_over(one_act_embed, output_embed), lambda: (one_act_embed, output_embed)), lambda: self.sequence_part_over(one_act_embed, output_embed))
      return i, i_len, one_act_embed, output_embed
    
    i = tf.constant(0, int_type)
    i_len = tf.shape(compute_tensor)[-1]
    one_act_embed = tf.zeros([1, num_units], float_type)
    output_embed = tf.zeros([0, num_units], float_type)
    _, _, one_act_embed, output_embed = tf.while_loop(compute_embed_cond, compute_embed_body, [i, i_len, one_act_embed, output_embed], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([1, num_units]), tf.TensorShape([None, num_units])])
    return output_embed


class DeepQ():
  
  def __init__(self, prefix):
    self.prefix = prefix
#     self.s_batch = tf.placeholder(float_type, [None, num_units], self.prefix + "s_batch")
#     self.a_batch = tf.placeholder(float_type, [None, num_units], self.prefix + "a_batch")
#     self.a_segment_batch = tf.placeholder(int_type, [None], self.prefix + "a_segment_batch")
#     self.policy_s_batch = tf.placeholder(float_type, [None, num_units], self.prefix + "policy_s_batch")
#     self.policy_actions_batch = tf.placeholder(float_type, [None, None, num_units], self.prefix + "policy_a_batch")
#     self.policy_actions_segment_batch = tf.placeholder(float_type, [None, num_units], self.prefix + "policy_a_segment_batch")
    
  def compute_q_value(self, s_batch, a_batch, a_segment_batch, r_branch_id_batch, r_segment_batch):
    
    def s_batch_cond(i, i_len, *_):
      return tf.less(i, i_len)
    
    def s_batch_body(i, i_len, a_seg_start, r_seg_start, q_v_batch):
      s_exmp = s_batch[i]
      a_seg_end = a_segment_batch[i]
      r_seg_end = r_segment_batch[i]
      
      def a_batch_cond(a, a_len, *_):
        return tf.less(a, a_len)
      
      def a_batch_body(a, a_len, a_q_v_batch):
        a_exmp = a_batch[a]
        
        def r_batch_cond(r, r_len, *_):
          return tf.less(r, r_len)
        
        def r_batch_body(r, r_len, r_q_v_batch):
          r_exmp_branch_id = tf.cast(r_branch_id_batch[r], int_type)
#           r_exmp_branch_influence = r_batch[1][r]
          with tf.variable_scope(name_or_scope="yyx_q_network", reuse=tf.AUTO_REUSE, dtype=float_type):
            w1 = tf.get_variable(self.prefix + "fusion1")[r_exmp_branch_id]
            w2 = tf.get_variable(self.prefix + "fusion2")[r_exmp_branch_id]
          q_val_im = tf.matmul(tf.expand_dims(tf.concat([s_exmp, a_exmp], axis=0), axis=0), w1)
          q_val = tf.matmul(tf.nn.tanh(q_val_im), w2)
          q_val = tf.squeeze(tf.nn.tanh(q_val))
          r_q_v_batch = tf.concat([r_q_v_batch, [q_val]], axis=0)
          return r+1, r_len, r_q_v_batch
        
        r_q_v_batch = tf.zeros([0], float_type)
        _, _, r_q_v_batch = tf.while_loop(r_batch_cond, r_batch_body, [r_seg_start, r_seg_end, r_q_v_batch], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])])
        a_q_v_batch = tf.concat([a_q_v_batch, r_q_v_batch], axis=0)
        return a+1, a_len, a_q_v_batch
      
      a_q_v_batch = tf.zeros([0], float_type)
      _, _, a_q_v_batch = tf.while_loop(a_batch_cond, a_batch_body, [a_seg_start, a_seg_end, a_q_v_batch], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])])
      q_v_batch = tf.concat([q_v_batch, a_q_v_batch], axis=0)
      return i+1, i_len, a_seg_end, r_seg_end, q_v_batch
    
    i = tf.constant(0, int_type)
    i_len = tf.shape(a_segment_batch)[-1]
    a_seg_start = tf.constant(0, int_type)
    r_seg_start = tf.constant(0, int_type)
    q_v_batch = tf.zeros([0], float_type)
    _, _, _, _, q_v_batch = tf.while_loop(s_batch_cond, s_batch_body, [i, i_len, a_seg_start, r_seg_start, q_v_batch], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])])
    return q_v_batch
    
#   def compute_q_value(self):
#     q_val_batch = self.compute_q_value_util(self.s_batch, self.a_batch)
#     self.q_batch_value = tf.identity(q_val_batch, name=self.prefix + "q_batch_value")
  
  def compute_next_policy_q_values(self, s_batch, a_batch, a_segment_batch, r_branch_id_batch, r_segment_batch):
#     actions_batch_embeds = tf.transpose(policy_actions_batch, perm=[1, 0, 2])
#      
#     def action_loop_cond(i, i_len, *_):
#       return tf.less(i, i_len)
#      
#     def action_loop_body(i, i_len, actions_q_batch_value):
#       q_batch_value = self.compute_q_value_util(policy_s_batch, actions_batch_embeds[i])
#       actions_q_batch_value = tf.concat([actions_q_batch_value, q_batch_value], axis=0)
#       return i+1, i_len, actions_q_batch_value
    
    q_val_batch = self.compute_q_value(s_batch, a_batch, a_segment_batch, r_branch_id_batch, r_segment_batch)
    
    def s_itr_cond(i, i_len, *_):
      return tf.less(i, i_len)
    
    def s_itr_body(i, i_len, accumulated_base, a_seg_start, r_seg_start, next_policy_q_vs):
      a_seg_end = a_segment_batch[i]
      r_seg_end = r_segment_batch[i]
      a_size = a_seg_end-a_seg_start
      r_size = r_seg_end-r_seg_start
      
      def r_itr_cond(r, r_len, *_):
        return tf.less(r, r_len)
      
      def r_itr_body(r, r_len, r_policy_q_vs):
        
        def a_itr_cond(a, a_len, *_):
          return tf.less(a, a_len)
        
        def a_itr_body(a, a_len, a_q_vs):
          a_q_vs = tf.concat([a_q_vs, [q_val_batch[accumulated_base+r+a*r_size]]], axis=0)
          return a+1, a_len, a_q_vs
        
        a_q_vs = tf.zeros([0], float_type)
        _, _, a_q_vs = tf.while_loop(a_itr_cond, a_itr_body, [tf.constant(0, int_type), a_seg_end-a_seg_start, a_q_vs])
        max_a_q = tf.reduce_max(a_q_vs)
        r_policy_q_vs = tf.concat([r_policy_q_vs, [max_a_q]], axis=0)
        return r+1, r_len, r_policy_q_vs
        
      accumulated_base = accumulated_base + a_size * r_size
      r_policy_q_vs = tf.zeros([0], float_type)
      _, _, r_policy_q_vs = tf.while_loop(r_itr_cond, r_itr_body, [tf.constant(0, int_type), r_seg_end-r_seg_start, r_policy_q_vs], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])])
      next_policy_q_vs = tf.concat([next_policy_q_vs, r_policy_q_vs], axis=0)
      return i+1, i_len, accumulated_base, a_seg_end, r_seg_end, next_policy_q_vs
    
    i = tf.constant(0, int_type)
    i_len = tf.shape(a_segment_batch)[-1]
    accumulated_base = tf.constant(0, int_type)
    a_seg_start = tf.constant(0, int_type)
    r_seg_start = tf.constant(0, int_type)
    next_policy_q_vs = tf.zeros([0], float_type)
    _, _, _, _, _, next_policy_q_vs = tf.while_loop(s_itr_cond, s_itr_body, [i, i_len, accumulated_base, a_seg_start, r_seg_start, next_policy_q_vs], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None])])
#   
#     def select_action(x):
#       start, end = x
#       part = tf.slice(q_val_batch, [start], [end - start])
#       return start + tf.argmax(part, axis=0, output_type=int_type)
#   
#     start_a_segment_batch = tf.concat([[0], tf.slice(a_segment_batch, [0], [tf.shape(a_segment_batch)[-1] - 1])], axis=0)
#     selected_actions = tf.map_fn(select_action, [start_a_segment_batch, a_segment_batch], (int_type))
    return next_policy_q_vs
  
#     i = tf.constant(0, int_type)
#     batch_size = tf.shape(policy_actions_batch)[0]
#     action_num = tf.shape(actions_batch_embeds)[0]
#     actions_q_batch_value = tf.zeros([0, batch_size], float_type)
#     _, _, actions_q_batch_value = tf.while_loop(action_loop_cond, action_loop_body, [i, action_num, actions_q_batch_value], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(None, None)])
#     batch_actions_q_value = tf.transpose(actions_q_batch_value, perm=[1, 0])
#     selected_actions = tf.arg_max(batch_actions_q_value, dimension=1)
#     return selected_actions
# 
#   def perform_policy(self):
#     selected_action_batch = self.perform_policy_util(self.s_batch, self.a_batch)
#     self.selected_action_batch = tf.identity(selected_action_batch, name=self.prefix + "selected_actions")
#     
#   def get_output_node_names(self):
#     return [self.prefix + "q_batch_value"]#, self.prefix + "selected_actions"


class QLearn():
  
  def __init__(self, sess, action_value, target_action_value):
    self.sess = sess
    self.action_value = action_value
    self.target_action_value = target_action_value
    self.adam = tf.train.AdamOptimizer()
    self.embed_computer = EmbedComputer()
    
    '''
    DenseObjectMatrix2D s_t_batch = new DenseObjectMatrix2D(2,0);
    DenseObjectMatrix1D s_t_segment_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix2D a_t_batch = new DenseObjectMatrix2D(2,0);
    DenseObjectMatrix1D a_t_segment_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix1D r_t_branch_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix1D r_t_segment_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix1D r_t_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix2D s_t_1_batch = new DenseObjectMatrix2D(2,0);
    DenseObjectMatrix1D s_t_1_segment_batch = new DenseObjectMatrix1D(0);
    DenseObjectMatrix2D s_t_1_actions_batch = new DenseObjectMatrix2D(2,0);
    DenseObjectMatrix1D s_t_1_actions_segment_batch = new DenseObjectMatrix1D(0);
    '''
    
    self.s_t_batch = tf.placeholder(int_type, [2, None], "s_t_batch")
    self.s_t_segment_batch = tf.placeholder(int_type, [None], "s_t_segment_batch")
    self.a_t_batch = tf.placeholder(int_type, [2, None], "a_t_batch")
    self.a_t_segment_batch = tf.placeholder(int_type, [None], "a_t_segment_batch")
    self.r_t_branch_batch = tf.placeholder(int_type, [None], "r_t_branch_batch")
    self.r_t_segment_batch = tf.placeholder(int_type, [None], "r_t_segment_batch")
    '''
    the data above this note is also used for predicting
    '''
    self.r_t_batch = tf.placeholder(float_type, [None], "r_t_batch")
    self.s_t_1_batch = tf.placeholder(int_type, [2, None], "s_t_1_batch")
    self.s_t_1_segment_batch = tf.placeholder(int_type, [None], "s_t_1_segment_batch")
    self.s_t_1_actions_batch = tf.placeholder(int_type, [2, None], "s_t_1_actions_batch")
    self.s_t_1_actions_segment_batch = tf.placeholder(int_type, [None], "s_t_1_actions_segment_batch")
    
  def predicting(self):
    s_t_embed_batch, a_t_embed_batch, a_t_embed_segment_batch = self.embed_computer.compute_states_actions_embed(self.s_t_batch, self.s_t_segment_batch, self.a_t_batch, self.a_t_segment_batch)
    self.predicted_q_values = self.action_value.compute_q_value(s_t_embed_batch, a_t_embed_batch, a_t_embed_segment_batch, self.r_t_branch_batch, self.r_t_segment_batch)
  
  def predicting_with_input(self, input_data):
    feed_dict = {
      self.s_t_batch : input_data["s_t_batch"],
      self.s_t_segment_batch : input_data["s_t_segment_batch"],
      self.a_t_batch : input_data["a_t_batch"],
      self.a_t_segment_batch : input_data["a_t_segment_batch"],
      self.r_t_branch_batch : input_data["r_t_branch_batch"],
      self.r_t_segment_batch : input_data["r_t_segment_batch"],
    }
    predicted_q_values = self.sess.run([self.predicted_q_values], feed_dict=feed_dict)
    predicted_q_values = predicted_q_values[0]
    output_data = {}
    a_start = 0
    r_start = 0
    accumulated_base = 0
    for s in range(len(input_data["s_t_segment_batch"])):
      a_end = input_data["a_t_segment_batch"][s]
      r_end = input_data["r_t_segment_batch"][s]
      a_length = a_end - a_start
      r_length = r_end - r_start
      s_data = {}
      output_data[str(s)] = s_data
      for a in range(a_length):
        r_data = []
        s_data[str(a)] = r_data
        for r in range(r_length):
          r_data.append(predicted_q_values[accumulated_base+r+a*r_length])
      a_start = a_end
      r_start = r_end
      accumulated_base = accumulated_base + a_length * r_length
    return output_data
    
  def learning(self):
    s_t_1_embed_batch, a_t_1_embed_batch, a_t_1_embed_segment_batch = self.embed_computer.compute_states_actions_embed(self.s_t_1_batch, self.s_t_1_segment_batch, self.s_t_1_actions_batch, self.s_t_1_actions_segment_batch)
    next_policy_q_values = self.target_action_value.compute_next_policy_q_values(s_t_1_embed_batch, a_t_1_embed_batch, a_t_1_embed_segment_batch, self.r_t_branch_batch, self.r_t_segment_batch)
#     def select_action(select_data):
#       return tf.squeeze(a_t_1_embed_batch[select_data])
#     selected_action_embeds = tf.map_fn(select_action, [selected_actions], dtype=(float_type))
    y_batch = self.r_t_batch + gamma * next_policy_q_values
#     self.target_action_value.compute_q_value(s_t_1_embed_batch, selected_action_embeds, tf.range(1, tf.shape(s_t_1_embed_batch)[0] + 1))
#     s_t_embed_batch, a_t_embed_batch, a_t_embed_segment_batch = self.embed_computer.compute_states_actions_embed(self.s_t_batch, self.s_t_segment_batch, self.a_t_batch, self.a_t_segment_batch)
#     action_value_q_vals = self.action_value.compute_q_value(s_t_embed_batch, a_t_embed_batch, a_t_embed_segment_batch, self.r_t_branch_batch, self.r_t_segment_batch)
    self.loss = tf.reduce_sum(tf.squared_difference(y_batch, self.predicted_q_values), name="q_learning_loss")
#     tf.losses.mean_squared_error()
    self.train = self.adam.minimize(self.loss, name="q_learning_train")
    
  def learning_with_input(self, input_data):
    feed_dict = {
      self.s_t_batch : input_data["s_t_batch"],
      self.s_t_segment_batch : input_data["s_t_segment_batch"],
      self.a_t_batch : input_data["a_t_batch"],
      self.a_t_segment_batch : input_data["a_t_segment_batch"],
      self.r_t_branch_batch : input_data["r_t_branch_batch"],
      self.r_t_segment_batch : input_data["r_t_segment_batch"],
      self.r_t_batch : input_data["r_t_batch"],
      self.s_t_1_batch : input_data["s_t_1_batch"],
      self.s_t_1_segment_batch : input_data["s_t_1_segment_batch"],
      self.s_t_1_actions_batch : input_data["s_t_1_actions_batch"],
      self.s_t_1_actions_segment_batch : input_data["s_t_1_actions_segment_batch"],
    }
    loss_val, _ = self.sess.run([self.loss, self.train], feed_dict=feed_dict)
    r_loss_val = np.asscalar(loss_val)
    return r_loss_val
    
#   def get_output_node_names(self):
#     return ["q_learning_loss", "q_learning_train"]
  

def recv_basic(the_socket):
  total_data = []
  while True:
    data = the_socket.recv(1024)    
    if not data: break
    total_data.extend(list(data))
  return total_data
  
  
def print_training_data(one_data):
#   print("s_t_batch:" + str(one_data["s_t_batch"]))
#   print("s_t_segment_batch:" + str(one_data["s_t_segment_batch"]))
#   print("a_t_batch:" + str(one_data["a_t_batch"]))
#   print("a_t_segment_batch:" + str(one_data["a_t_segment_batch"]))
#   print("r_t_batch:" + str(one_data["r_t_batch"]))
#   print("s_t_1_actions_batch:" + str(one_data["s_t_1_actions_batch"]))
#   print("s_t_1_actions_segment_batch:" + str(one_data["s_t_1_actions_segment_batch"]))
#   print("s_t_1_batch:" + str(one_data["s_t_1_batch"]))
#   print("s_t_1_segment_batch:" + str(one_data["s_t_1_segment_batch"]))
  pass

  
if __name__ == '__main__':
  with tf.Session() as sess:
    variable_initialize()
    action_value = DeepQ(action_value_prefix)
#     action_value.compute_q_value()
#     action_value.perform_policy()
    target_action_value = DeepQ(target_action_value_prefix)
#     target_action_value.compute_q_value()
#     target_action_value.perform_policy()
    q_learn = QLearn(sess, action_value, target_action_value)
    q_learn.predicting()
    q_learn.learning()
    '''
    write model to file
    '''
    output_node_names = []
#     output_node_names = output_node_names + action_value.get_output_node_names()
#     output_node_names = output_node_names + target_action_value.get_output_node_names()
#     output_node_names = output_node_names + q_learn.get_output_node_names()
    sess.run(tf.global_variables_initializer())
#     output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=output_node_names)
#     with tf.gfile.FastGFile('refined_deep_q.pb', mode='wb') as f:
#       f.write(output_graph_def.SerializeToString())
    address = ('127.0.0.1', 31500)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # s = socket.socket()
    s.bind(address)
    s.listen(5)
    while True:
      conn, addr = s.accept()
      print("Connected by " + str(addr))
      json_raw_data = recv_basic(conn)
      one_data = json.loads(bytes(json_raw_data))
      if one_data == "stop": break
#       print("one_data:" + str(one_data))
      
      if "learning" in one_data:
        assert "learning" in one_data
        one_training_data = one_data["learning"]
        print_training_data(one_training_data)
        return_v = q_learn.learning_with_input(one_training_data)
      else:
        assert "predicting" in one_data
        one_predicting_data = one_data["predicting"]
        return_v = q_learn.predicting_with_input(one_predicting_data)
        
      '''
      send running result to Java
      '''
      s_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      s_c.connect(('127.0.0.1', 41500))
      s_c.send(json.dumps(return_v).encode())
      s_c.close()
    s.close()
  '''
  x: [batch, actions, num_units]
  y: [batch] # index of selected actions
  o: [batch, 1, num_units]
  
  o = tf.map_fn(lambda p: p[0][p[1]], (x, y), (tf.float32))
  '''
  
