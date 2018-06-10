"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np
import imageio
import time
import os
import save_images
import custom_layers

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../tensorflow_learning/MNIST/MNIST_data/", one_hot=True)

initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z):
    fcW1 = tf.get_variable('fcW1', [25, 1024], initializer=initializer)
    fcb1 = tf.get_variable('fcb1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = tf.nn.sigmoid(tf.matmul(z, fcW1) + fcb1)

    W2 = tf.get_variable('W2', [1024, 8 * 8 * 64], initializer=initializer)
    b2 = tf.get_variable('b2', [8 * 8 * 64], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)
    f2 = tf.reshape(f2, [-1, 8, 8, 64])

    W_conv1 = tf.get_variable('W_conv1', [5, 5, 32, 64], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [16, 16, 32], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(f2, W_conv1, [100, 16, 16, 32], [1, 2, 2, 1]) + b_conv1)

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 1, 32], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [32, 32, 1], initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.tanh(tf.nn.conv2d_transpose(conv1, W_conv2, [100, 32, 32, 1], [1, 2, 2, 1]) + b_conv2)

    return conv2

def discriminator(x):
    W_conv1 = tf.get_variable('W_conv1', [5, 5, 1, 32], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [32], initializer=tf.constant_initializer(0.0))

    h_conv1 = custom_layers.lrelu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [64], initializer=tf.constant_initializer(0.0))

    h_conv2 = custom_layers.lrelu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

    W1 = tf.get_variable('W1', [8*8*64, 1024], initializer=initializer)
    b1 = tf.get_variable('b1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = custom_layers.lrelu(tf.matmul(h_pool2_flat, W1) + b1)

    W2 = tf.get_variable('W2', [1024, 1], initializer=initializer)
    b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

    f2 = tf.matmul(f1, W2) + b2

    return f2

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, [100, 25])
    G = generator(z)

x = tf.placeholder(tf.float32, [None, 32, 32, 1])
with tf.variable_scope('D'):
    Dx = discriminator(x)
with tf.variable_scope('D', reuse=True):
    Dg = discriminator(G)

def log(x):
    """
    Sometimes the discriminator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimization.
    """
    return tf.log(tf.maximum(x, 1e-5))

eps = tf.random_uniform([100, 32, 32, 1], minval=-1., maxval=1.)
X_inter = eps*x + (1. - eps)*G
with tf.variable_scope('D', reuse=True):
    grad = tf.gradients(discriminator(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = 10 * tf.reduce_mean((grad_norm - 1)**2)

loss_d = tf.reduce_mean(Dg) - tf.reduce_mean(Dx) + grad_pen #This optimizes the discriminator.
loss_g = -tf.reduce_mean(Dg) #This optimizes the generator.


vars = tf.trainable_variables()
for v in vars:
    print(v.name)
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

d_opt = tf.train.AdamOptimizer(1e-4).minimize(loss_d, var_list=d_params)
g_opt = tf.train.AdamOptimizer(1e-4).minimize(loss_g, var_list=g_params)

with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    start_time = time.time()
    sample_directory = 'generated_images/mnist_wasserstein/{}'.format(start_time)
    for step in range(100000):
        # update discriminator
        for i in range(10):
            mnist_images, _ = mnist.train.next_batch(100)
            mnist_images = (np.reshape(mnist_images, [100, 28, 28, 1]) - 0.5) * 2.0
            mnist_images = np.lib.pad(mnist_images, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1))
            input_noise = np.random.rand(100, 25)
            loss_d_thingy, _ = session.run([loss_d, d_opt], {x: mnist_images, z: input_noise})

        # update generator
        input_noise = np.random.rand(100, 25)
        loss_g_thingy, _ = session.run([loss_g, g_opt], {z: input_noise})

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))

        if step % 100 == 0:
            input_noise = np.random.rand(100, 25)
            image = session.run(G, {z: input_noise})
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images.save_images(np.reshape(image, [100, 32, 32, 1]), [10, 10], sample_directory + '/fig{}.png'.format(step))
