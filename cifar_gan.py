"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np
import imageio
import time

import os
from os import listdir
from os.path import isfile, join

def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=1)
  image_normalized = tf.image.convert_image_dtype(image_decoded, tf.float32)
  image_cropped = tf.image.crop_to_bounding_box(image_normalized, 4, 4, 32, 32)
  return image_cropped

def load_images(batch_size):
    dataset = tf.data.Dataset.list_files("E:\\cifar10\\train\\*")
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(batch_size)
    # dataset = dataset.shuffle(10000)
    return dataset.make_one_shot_iterator()


#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))

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
    fcW1 = tf.get_variable('fcW1', [25, 1024], initializer=initializer)
    fcb1 = tf.get_variable('fcb1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = tf.nn.sigmoid(tf.matmul(z, fcW1) + fcb1)
    f1 = tf.layers.batch_normalization(f1)

    W2 = tf.get_variable('W2', [1024, 8 * 8 * 64], initializer=initializer)
    b2 = tf.get_variable('b2', [8 * 8 * 64], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)
    f2 = tf.layers.batch_normalization(f2)
    f2 = tf.reshape(f2, [-1, 8, 8, 64])

    W_conv1 = tf.get_variable('W_conv1', [5, 5, 32, 64], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [16, 16, 32], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(f2, W_conv1, [100, 16, 16, 32], [1, 2, 2, 1]) + b_conv1)
    conv1 = tf.layers.batch_normalization(conv1)

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 1, 32], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [32, 32, 1], initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(conv1, W_conv2, [100, 32, 32, 1], [1, 2, 2, 1]) + b_conv2)

    return conv2

def minibatch_layer(input, num_kernels=5, kernel_dim=3):
    mb_fc_W = tf.get_variable('mb_fc_W', [input.get_shape()[1], num_kernels * kernel_dim], initializer=initializer)
    mb_fc_b = tf.get_variable('mb_fc_b', [num_kernels * kernel_dim], initializer=tf.constant_initializer(0.0))

    mb_fc = tf.matmul(input, mb_fc_W) + mb_fc_b

    activation = tf.reshape(mb_fc, [-1, num_kernels, kernel_dim])
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def discriminator(x):
    W_conv1 = tf.get_variable('W_conv1', [5, 5, 1, 32], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [32], initializer=tf.constant_initializer(0.0))

    h_conv1 = lrelu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [64], initializer=tf.constant_initializer(0.0))

    h_conv2 = lrelu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

    minibatch = minibatch_layer(h_pool2_flat, 10, 6)

    W1 = tf.get_variable('W1', [8*8*64 + 10, 1024], initializer=initializer)
    b1 = tf.get_variable('b1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = lrelu(tf.matmul(minibatch, W1) + b1)

    W2 = tf.get_variable('W2', [1024, 1], initializer=initializer)
    b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)

    return f2

with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, [100, 25])
    G = generator(z)

x = load_images(100).get_next()
with tf.variable_scope('D'):
    Dx = discriminator(x)
with tf.variable_scope('D', reuse=True):
    Dg = discriminator(G)

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

    start_time = time.time()
    sample_directory = 'generated_images/cifar/{}'.format(start_time)
    for step in range(10000):
        # update discriminator
        input_noise = np.random.rand(100, 25)
        loss_d_thingy, _ = session.run([loss_d, d_opt], {z: input_noise})

        # update generator
        for i in range(10):
            input_noise = np.random.rand(100, 25)
            loss_g_thingy, _ = session.run([loss_g, g_opt], {z: input_noise})

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))

        if step % 100 == 0:
            input_noise = np.random.rand(100, 25)
            gen_image = session.run(G, {z: input_noise})
            real_image = session.run(x)
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images(np.reshape(gen_image, [100, 32, 32]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            save_images(np.reshape(real_image, [100, 32, 32]), [10, 10], sample_directory + '/{}real.png'.format(step))