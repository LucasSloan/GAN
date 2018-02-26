"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../tensorflow_learning/MNIST/MNIST_data/", one_hot=True)


def generator(z):
    W1 = tf.get_variable('W1', [25, 784], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('b1', [784], initializer=tf.constant_initializer(0.0))
    
    mid = tf.nn.relu(tf.matmul(z, W1) + b1)

    W2 = tf.get_variable('W2', [784, 784], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('b2', [784], initializer=tf.constant_initializer(0.0))

    return tf.nn.relu(tf.matmul(mid, W2) + b2)

def discriminator(x):
    W1 = tf.get_variable('W1', [784, 25], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('b1', [25], initializer=tf.constant_initializer(0.0))
    
    mid = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.get_variable('W2', [25, 1], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

    return tf.sigmoid(tf.matmul(mid, W2) + b2)

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, [100, 25])
    G = generator(z)

x = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope('D'):
    D1 = discriminator(x)
with tf.variable_scope('D', reuse=True):
    D2 = discriminator(G)

def log(x):
    """
    Sometimes the discriminator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimization.
    """
    return tf.log(tf.maximum(x, 1e-5))


loss_d = tf.reduce_mean(-log(D1) - log(1 - D2))
loss_g = tf.reduce_mean(-log(D2))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

d_opt = tf.train.AdamOptimizer(1e4).minimize(loss_d, var_list=d_params)
g_opt = tf.train.AdamOptimizer(1e4).minimize(loss_g, var_list=g_params)

with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    for step in range(100):
        # update discriminator
        mnist_images, _ = mnist.train.next_batch(100)
        input_noise = np.random.rand(100, 25)
        loss_d_thingy, _ = session.run([loss_d, d_opt], {x: mnist_images, z: input_noise})

        # update generator
        input_noise = np.random.rand(100, 25)
        loss_g_thingy, _ = session.run([loss_g, g_opt], {z: input_noise})

        if step % 1 == 0:
            print('{}: {:.4f}\t{:.4f}'.format(step, loss_d_thingy, loss_g_thingy))