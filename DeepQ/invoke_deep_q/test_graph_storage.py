import tensorflow as tf


x = tf.placeholder("float", shape=[None, 784], name='input_x')
y_ = tf.placeholder("float", shape=[None, 10], name='input_y')

W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')
tf.initialize_all_variables().run()
y = tf.nn.softmax(tf.matmul(x,W)+b, name='softmax')

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, name='train_step')
# train_step.run(feed_dict={x:input_x, y_:input_y})
