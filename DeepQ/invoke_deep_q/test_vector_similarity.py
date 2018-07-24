import tensorflow as tf
from invoke_deep_q.invoke import float_type


c = [0.1, 0.5]
b = [0.1, 0.3]
a = [0.2, 0.2]
c_norm = tf.nn.l2_normalize(c)
b_norm = tf.nn.l2_normalize(b)
a_norm = tf.nn.l2_normalize(a)
sess = tf.InteractiveSession()
print(sess.run([tf.tensordot(a_norm, b_norm, axes=1), tf.tensordot(a_norm, c_norm, axes=1)]))
sess.close()


ll = []
n = 8
test_var = tf.get_variable("test_var", [n, n], float_type, initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=None, dtype=float_type))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(n):
  i_ll = []
  ll.append(i_ll)
  for j in range(n):
    i_ll.append(tf.tensordot(tf.nn.l2_normalize(test_var[i]), tf.nn.l2_normalize(test_var[j]), axes=1))
ll_val = sess.run([ll])[0]
for i in range(len(ll_val)):
  print(' '.join(['%6f' % x for x in ll_val[i]]))
sess.close()



