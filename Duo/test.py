import tensorflow as tf
import numpy as np

x = np.random.randn(1, 28, 28, 3)
x = tf.convert_to_tensor(x, dtype=tf.float64)
x = tf.Variable(x, name='x')

conv_filter = tf.get_variable('weights_cpu', [11, 11, 3, 20],
                                      initializer=tf.constant_initializer(1),
                                      dtype=tf.float64)
output_tensorflow = tf.nn.convolution(x, filter=conv_filter,
                                              strides=[4, 4], padding='VALID')
with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
    with tf.Session() as sess:
        x = np.random.randn(1, 14, 14, 3)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        sess.run(tf.global_variables_initializer())
        sess.run(output_tensorflow)

        tf.get_variable_scope().reuse_variables()
