import tensorflow as tf
import numpy as np
import imageio
import time
import save_images
import custom_layers
import os

def parse_images(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_normalized = 2.0 * tf.image.convert_image_dtype(image_decoded, tf.float32) - 1.0
  image_resized = tf.image.resize_images(image_normalized, [64, 64])
  image_flipped = tf.image.flip_left_right(image_resized)
  return image_flipped

def load_images(batch_size):
    image_files_dataset = tf.data.Dataset.list_files("E:\\LSUN\\train\\*")
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    dataset = image_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(5000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()

initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z):
    fcW1 = tf.get_variable('fcW1', [100, 1024], initializer=initializer)
    fcb1 = tf.get_variable('fcb1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = tf.nn.sigmoid(tf.matmul(z, fcW1) + fcb1)
    f1 = tf.layers.batch_normalization(f1, fused=True)

    W2 = tf.get_variable('W2', [1024, 8 * 8 * 128], initializer=initializer)
    b2 = tf.get_variable('b2', [8 * 8 * 128], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)
    f2 = tf.layers.batch_normalization(f2, fused=True)
    f2 = tf.reshape(f2, [-1, 8, 8, 128])

    W_conv1 = tf.get_variable('W_conv1', [5, 5, 64, 128], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [16, 16, 64], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(f2, W_conv1, [100, 16, 16, 64], [1, 2, 2, 1]) + b_conv1)
    conv1 = tf.layers.batch_normalization(conv1, fused=True)

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [32, 32, 1], initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(conv1, W_conv2, [100, 32, 32, 32], [1, 2, 2, 1]) + b_conv2)
    conv2 = tf.layers.batch_normalization(conv2, fused=True)

    W_conv3 = tf.get_variable('W_conv3', [5, 5, 3, 32], initializer=initializer)
    b_conv3 = tf.get_variable('b_conv3', [64, 64, 3], initializer=tf.constant_initializer(0.0))

    conv3 = tf.nn.tanh(tf.nn.conv2d_transpose(conv2, W_conv3, [100, 64, 64, 3], [1, 2, 2, 1]) + b_conv3)

    return conv3

def discriminator(x):
    W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 32], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [32], initializer=tf.constant_initializer(0.0))

    h_conv1 = custom_layers.lrelu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [64], initializer=tf.constant_initializer(0.0))

    h_conv2 = custom_layers.lrelu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv3 = tf.get_variable('W_conv3', [5, 5, 64, 128], initializer=initializer)
    b_conv3 = tf.get_variable('b_conv3', [128], initializer=tf.constant_initializer(0.0))

    h_conv3 = custom_layers.lrelu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*128])

    minibatch = custom_layers.minibatch_layer(h_pool3_flat, initializer, 100, 5)

    # W1 = tf.get_variable('W1', [8*8*128 + 10, 2048], initializer=initializer)
    # b1 = tf.get_variable('b1', [2048], initializer=tf.constant_initializer(0.0))
    
    # f1 = lrelu(tf.matmul(minibatch, W1) + b1)

    W2 = tf.get_variable('W2', [8*8*128 + 100, 1], initializer=initializer)
    b2 = tf.get_variable('b2', [1], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(minibatch, W2) + b2)

    return f2

with tf.variable_scope('G'):
    z = tf.random_uniform([100, 100])
    G = generator(z)

x = load_images(100).get_next()
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

    start_time = time.time()
    previous_step_time = time.time()
    sample_directory = 'generated_images/lsun/{}'.format(start_time)
    for step in range(1000000):
        # update discriminator
        loss_d_thingy, _ = session.run([loss_d, d_opt])

        # update generator
        for i in range(5):
            loss_g_thingy, _ = session.run([loss_g, g_opt])

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))

        if step % 100 == 0:
            current_step_time = time.time()
            print('{}: previous 100 steps took {:.4f}s'.format(step, current_step_time - previous_step_time))
            previous_step_time = current_step_time
            
        if step % 1000 == 0:
            gen_image = session.run(G)
            real_image = session.run(x)
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images.save_images(np.reshape(gen_image, [100, 64, 64, 3]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            save_images.save_images(np.reshape(real_image, [100, 64, 64, 3]), [10, 10], sample_directory + '/{}real.png'.format(step))