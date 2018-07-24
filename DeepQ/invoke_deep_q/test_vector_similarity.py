import tensorflow as tf

c = [0.1, 0.5]
b = [0.1, 0.3]
a = [0.2, 0.2]
c_norm = tf.nn.l2_normalize(c)
b_norm = tf.nn.l2_normalize(b)
a_norm = tf.nn.l2_normalize(a)
sess = tf.InteractiveSession()
print(sess.run([tf.tensordot(a_norm, b_norm, axes=1), tf.tensordot(a_norm, c_norm, axes=1)]))
sess.close()

