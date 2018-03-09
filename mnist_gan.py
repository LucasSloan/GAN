"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np
import imageio

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../tensorflow_learning/MNIST/MNIST_data/", one_hot=True)

#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z):
    W1 = tf.get_variable('W1', [25, 7840], initializer=initializer)
    b1 = tf.get_variable('b1', [7840], initializer=tf.constant_initializer(0.0))
    
    f1 = tf.nn.sigmoid(tf.matmul(z, W1) + b1)
    f1 = tf.reshape(f1, [-1, 28, 28, 10])

    # W2 = tf.get_variable('W2', [3500, 3500], initializer=initializer)
    # b2 = tf.get_variable('b2', [3500], initializer=tf.constant_initializer(0.0))

    # f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)

    # W3 = tf.get_variable('W3', [3500, 784], initializer=initializer)
    # b3 = tf.get_variable('b3', [784], initializer=tf.constant_initializer(0.0))

    # f3 = tf.nn.tanh(tf.matmul(f2, W3) + b3)

    W_conv1 = tf.get_variable('W2', [5, 5, 1, 10], initializer=initializer)
    b_conv1 = tf.get_variable('b2', [28, 28, 1], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.tanh(tf.nn.conv2d_transpose(f1, W_conv1, [100, 28, 28, 1], [1, 1, 1, 1]) + b_conv1)

    # W_conv2 = tf.get_variable('W3', [5, 5, 1, 32], initializer=initializer)
    # b_conv2 = tf.get_variable('b3', [28, 28, 1], initializer=tf.constant_initializer(0.0))

    # conv2 = tf.nn.tanh(tf.nn.conv2d_transpose(conv1, W_conv2, [100, 28, 28, 1], [1, 1, 1, 1]) + b_conv2)
    conv1 = tf.reshape(conv1, [-1, 784])

    return conv1

def discriminator(x):
    W1 = tf.get_variable('W1', [784, 3500], initializer=initializer)
    b1 = tf.get_variable('b1', [3500], initializer=tf.constant_initializer(0.0))
    
    f1 = lrelu(tf.matmul(x, W1) + b1)

    W2 = tf.get_variable('W2', [3500, 3500], initializer=initializer)
    b2 = tf.get_variable('b2', [3500], initializer=tf.constant_initializer(0.0))

    f2 = lrelu(tf.matmul(f1, W2) + b2)

    W3 = tf.get_variable('W3', [3500, 1], initializer=initializer)
    b3 = tf.get_variable('b3', [1], initializer=tf.constant_initializer(0.0))

    f3 = tf.nn.sigmoid(tf.matmul(f2, W3) + b3)

    return f3

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, [100, 25])
    G = generator(z)

x = tf.placeholder(tf.float32, [None, 784])
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


loss_d = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
loss_g = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.


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

    for step in range(10000):
        # update discriminator
        mnist_images, _ = mnist.train.next_batch(100)
        mnist_images = (mnist_images - 0.5) * 2.0
        # print(mnist_images[0])
        input_noise = np.random.rand(100, 25)
        # print(input_noise)
        loss_d_thingy, _, dx, dg = session.run([loss_d, d_opt, Dx, Dg], {x: mnist_images, z: input_noise})

        # print("Dx!!!!!!!!")
        # print(dx[:9])
        # print("Dg!!!!!!!!")
        # print(dg[:9])
        # update generator
        for i in range(7):
            input_noise = np.random.rand(100, 25)
            loss_g_thingy, _ = session.run([loss_g, g_opt], {z: input_noise})

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))

        if step % 100 == 0:
            input_noise = np.random.rand(100, 25)
            image = session.run(G, {z: input_noise})
            # print(image[0])
            save_images(np.reshape(image, [100, 28, 28]), [10, 10], 'generated_images/fig{}.png'.format(step))