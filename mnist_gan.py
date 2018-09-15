"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np
import imageio
import time
import os
import save_images
import custom_layers

def parse_images(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=1)
  image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 32, 32)
  image_normalized = 2.0 * tf.image.convert_image_dtype(image_resized, tf.float32) - 1.0
  return image_normalized

def load_images(batch_size):
    image_files_dataset = tf.data.Dataset.list_files("E:\\mnist\\train\\*\\*")
    image_files_dataset = image_files_dataset.concatenate(tf.data.Dataset.list_files("E:\\mnist\\test\\*\\*"))
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    dataset = image_dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()


initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z, training):
    f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
    f1 = tf.layers.batch_normalization(f1, training=training)

    f2 = tf.layers.dense(f1, 8 * 8 * 64, tf.nn.leaky_relu)
    f2 = tf.reshape(f2, [-1, 8, 8, 64])
    f2 = tf.layers.batch_normalization(f2, training=training)

    conv1 = tf.layers.conv2d_transpose(f2, 32, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.leaky_relu)
    conv1 = tf.layers.batch_normalization(conv1, training=training)

    conv2 = tf.layers.conv2d_transpose(conv1, 1, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.tanh)

    return conv2

def discriminator(x, training):
    h_conv1 = tf.layers.conv2d(x, 32, [5, 5], strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
    # h_conv1 = tf.layers.batch_normalization(h_conv1, training=training)

    h_conv2 = tf.layers.conv2d(h_conv1, 64, [5, 5], strides=(2, 2), padding='same', activation=tf.nn.leaky_relu)
    h_conv2_flat = tf.reshape(h_conv2, [-1, 8*8*64])
    # h_conv2_flat = tf.layers.batch_normalization(h_conv2_flat, training=training)

    f1 = tf.layers.dense(h_conv2_flat, 1024, tf.nn.leaky_relu)
    # f1 = tf.layers.batch_normalization(f1, training=training)

    f2 = tf.layers.dense(f1, 1, tf.nn.sigmoid)

    return f2

training = tf.placeholder(tf.bool)

with tf.variable_scope('G'):
    z = tf.random_uniform([100, 100])
    G = generator(z, training)

x = load_images(100).get_next()
with tf.variable_scope('D'):
    Dx = discriminator(x, training)
with tf.variable_scope('D', reuse=True):
    Dg = discriminator(G, training)

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

    previous_step_time = time.time()
    start_time = time.time()
    sample_directory = 'generated_images/mnist/{}'.format(start_time)
    for step in range(1, 100001):
        # update discriminator
        for _ in range(1):
            loss_d_thingy, _ = session.run([loss_d, d_opt], {training: True})

        # update generator
        for _ in range(1):
            loss_g_thingy, _ = session.run([loss_g, g_opt], {training: True})

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))

        if step % 100 == 0:
            current_step_time = time.time()
            print('{}: previous 100 steps took {:.4f}s'.format(step, current_step_time - previous_step_time))
            previous_step_time = current_step_time

        if step % 1000 == 0:
            gen_image = session.run(G, {training: True})
            real_image = session.run(x, {training: True})
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images.save_images(np.reshape(gen_image, [100, 32, 32, 1]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            save_images.save_images(np.reshape(real_image, [100, 32, 32, 1]), [10, 10], sample_directory + '/{}real.png'.format(step))
