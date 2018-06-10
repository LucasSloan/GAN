import tensorflow as tf

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def minibatch_layer(input, initializer, num_kernels=5, kernel_dim=3):
    mb_fc_W = tf.get_variable('mb_fc_W', [input.get_shape()[1], num_kernels * kernel_dim], initializer=initializer)
    mb_fc_b = tf.get_variable('mb_fc_b', [num_kernels * kernel_dim], initializer=tf.constant_initializer(0.0))

    mb_fc = tf.matmul(input, mb_fc_W) + mb_fc_b

    activation = tf.reshape(mb_fc, [-1, num_kernels, kernel_dim])
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
