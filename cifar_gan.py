"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np
import time

import os

import save_images

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def parse_images(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_flipped = tf.image.random_flip_left_right(image_decoded)
  # image_rand_bright = tf.image.random_brightness(image_flipped, 0.3)
  # image_rand_contrast = tf.image.random_contrast(image_rand_bright, 0.85, 1.15)
  # image_rand_hue = tf.image.random_hue(image_rand_contrast, 0.15)
  # image_rand_sat = tf.image.random_saturation(image_rand_hue, 0.85, 1.15)
  image_normalized = 2.0 * tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
#   image_noisy = gaussian_noise_layer(image_normalized, 0.05)
#   image_clipped = tf.clip_by_value(image_noisy, 0.0, 1.0)
  return image_normalized

def text_to_one_hot(text_label):
    int_label = tf.string_to_number(text_label, out_type=tf.int32)
    return tf.one_hot(int_label, 11)


def load_images_and_labels(batch_size):
    image_files_dataset = tf.data.Dataset.list_files("E:\\cifar10\\train\\*")
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    label_lines_dataset = tf.data.TextLineDataset(["E:\\cifar10\\Train_cntk_text.txt"])
    label_dataset = label_lines_dataset.map(text_to_one_hot)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()


#This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)



initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z):
    fcW1 = tf.get_variable('fcW1', [25, 1024], initializer=initializer)
    fcb1 = tf.get_variable('fcb1', [1024], initializer=tf.constant_initializer(0.0))
    
    f1 = tf.nn.sigmoid(tf.matmul(z, fcW1) + fcb1)
    f1 = tf.layers.batch_normalization(f1, fused=True)

    W2 = tf.get_variable('W2', [1024, 4 * 4 * 128], initializer=initializer)
    b2 = tf.get_variable('b2', [4 * 4 * 128], initializer=tf.constant_initializer(0.0))

    f2 = tf.nn.sigmoid(tf.matmul(f1, W2) + b2)
    f2 = tf.layers.batch_normalization(f2, fused=True)
    f2 = tf.reshape(f2, [-1, 4, 4, 128])

    W_conv1 = tf.get_variable('W_conv1', [5, 5, 64, 128], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [8, 8, 64], initializer=tf.constant_initializer(0.0))

    conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(f2, W_conv1, [100, 8, 8, 64], [1, 2, 2, 1]) + b_conv1)
    conv1 = tf.layers.batch_normalization(conv1, fused=True)

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [16, 16, 1], initializer=tf.constant_initializer(0.0))

    conv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(conv1, W_conv2, [100, 16, 16, 32], [1, 2, 2, 1]) + b_conv2)
    conv2 = tf.layers.batch_normalization(conv2, fused=True)

    W_conv3 = tf.get_variable('W_conv3', [5, 5, 3, 32], initializer=initializer)
    b_conv3 = tf.get_variable('b_conv3', [32, 32, 3], initializer=tf.constant_initializer(0.0))

    conv3 = tf.nn.tanh(tf.nn.conv2d_transpose(conv2, W_conv3, [100, 32, 32, 3], [1, 2, 2, 1]) + b_conv3)

    return conv3

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
    W_conv1 = tf.get_variable('W_conv1', [5, 5, 3, 32], initializer=initializer)
    b_conv1 = tf.get_variable('b_conv1', [32], initializer=tf.constant_initializer(0.0))

    h_conv1 = lrelu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.get_variable('W_conv2', [5, 5, 32, 64], initializer=initializer)
    b_conv2 = tf.get_variable('b_conv2', [64], initializer=tf.constant_initializer(0.0))

    h_conv2 = lrelu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv3 = tf.get_variable('W_conv3', [5, 5, 64, 128], initializer=initializer)
    b_conv3 = tf.get_variable('b_conv3', [128], initializer=tf.constant_initializer(0.0))

    h_conv3 = lrelu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])

    minibatch = minibatch_layer(h_pool3_flat, 100, 5)

    # W1 = tf.get_variable('W1', [4*4*128 + 10, 2048], initializer=initializer)
    # b1 = tf.get_variable('b1', [2048], initializer=tf.constant_initializer(0.0))
    
    # f1 = lrelu(tf.matmul(minibatch, W1) + b1)

    W2 = tf.get_variable('W2', [4*4*128 + 100, 11], initializer=initializer)
    b2 = tf.get_variable('b2', [11], initializer=tf.constant_initializer(0.0))

    f2 = tf.matmul(minibatch, W2) + b2

    return f2

with tf.variable_scope('G'):
    z = tf.random_uniform([100, 25])
    G = generator(z)

images_and_labels = load_images_and_labels(100).get_next()
x = images_and_labels[0]
yx = images_and_labels[1]
yg = tf.reshape(tf.tile(tf.one_hot(10, 11), [100]), [100, 11])

with tf.variable_scope('D'):
    Dx = discriminator(x)
with tf.variable_scope('D', reuse=True):
    Dg = discriminator(G)

# loss_d = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yx, logits=Dx) + tf.nn.softmax_cross_entropy_with_logits(labels=yg, logits=Dg)) #This optimizes the discriminator.
# loss_g = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.
loss_g = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yg, logits=Dg)) #This optimizes the generator.

correct_prediction = tf.equal(tf.argmax(Dg, 1), tf.argmax(yg, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
    sample_directory = 'generated_images/cifar/{}'.format(start_time)
    for step in range(1000000):
        # update discriminator
        loss_d_thingy, _ = session.run([loss_d, d_opt])

        # update generator
        for i in range(15):
            loss_g_thingy, _ = session.run([loss_g, g_opt])

        if step % 100 == 0:
            train_accuracy = session.run([accuracy])
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, loss_d_thingy, loss_g_thingy))
            print('train accuracy: {}'.format(train_accuracy))

        if step % 100 == 0:
            gen_image = session.run(G)
            real_image = session.run(x)
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            save_images.save_images(np.reshape(gen_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            save_images.save_images(np.reshape(real_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}real.png'.format(step))

        if step % 100 == 0:
            current_step_time = time.time()
            print('{}: previous 100 steps took {:.4f}s'.format(step, current_step_time - previous_step_time))
            previous_step_time = current_step_time
