"""Basic GAN to generate mnist images."""
import tensorflow as tf
import numpy as np

import time
import os
import sys
import shutil

import save_images
import custom_layers
import ops
import resnet_architecture
import consts

# Constants
CIFAR_CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
USE_SN = True
LABEL_BASED_DISCRIMINATOR = False
OUTPUT_REAL_IMAGES = False
DATA_DIR = "/home/lucas/training_data/cifar10/"

def parse_images(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=3)
  image_flipped = tf.image.random_flip_left_right(image_decoded)
  image_normalized = 2.0 * tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
  return image_normalized

def text_to_index(text_label):
    return tf.string_to_number(text_label, out_type=tf.int32)

def text_to_one_hot(text_label):
    int_label = tf.string_to_number(text_label, out_type=tf.int32)
    return tf.one_hot(int_label, 11)


def load_images_and_labels(batch_size):
    image_files_dataset = tf.data.Dataset.list_files(DATA_DIR + "train/*", shuffle=False)
    image_files_dataset = image_files_dataset.concatenate(tf.data.Dataset.list_files(DATA_DIR + "test/*", shuffle=False))
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=32)

    label_lines_dataset = tf.data.TextLineDataset([DATA_DIR + "Train_cntk_text.txt", DATA_DIR + "Test_cntk_text.txt"])
    label_dataset = label_lines_dataset.map(text_to_one_hot)
    index_dataset = label_lines_dataset.map(text_to_index)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset, index_dataset))

    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache()
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()


initializer = tf.truncated_normal_initializer(stddev=0.02)


def generator(z):
    f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
    f1 = tf.layers.batch_normalization(f1, training=True)

    f2 = tf.layers.dense(f1, 4*4*128, tf.nn.leaky_relu)
    f2 = tf.reshape(f2, [-1, 4, 4, 128])
    f2 = tf.layers.batch_normalization(f2, training=True)

    conv1 = tf.layers.conv2d_transpose(f2, 32, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.leaky_relu)
    conv1 = tf.layers.batch_normalization(conv1, training=True)

    conv2 = tf.layers.conv2d_transpose(conv1, 32, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.leaky_relu)
    conv2 = tf.layers.batch_normalization(conv2, training=True)

    conv3 = tf.layers.conv2d_transpose(conv2, 3, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.tanh)

    return conv3

def residual_block(input, channels, downsample, name):
    shortcut = input
    stride = 1
    if downsample:
        stride = 2
        shortcut = ops.conv2d(input, channels, 1, 1, stride, stride, name=name + "_shortcut", use_sn=USE_SN)
        # shortcut = tf.layers.batch_normalization(shortcut, training=True)
        
    conv1 = ops.conv2d(input, channels, 3, 3, 1, 1, name=name + "_conv1", use_sn=USE_SN)
    # conv1 = tf.layers.batch_normalization(conv1, training=True)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = ops.conv2d(conv1, channels, 3, 3, stride, stride, name=name + "_conv2", use_sn=USE_SN)
    # conv2 = tf.layers.batch_normalization(conv2, training=True)


    conv2 += shortcut
    output = tf.nn.leaky_relu(conv2)

    return output

def discriminator(x):
    # h_conv1 = tf.nn.leaky_relu(ops.conv2d(x, 32, 5, 5, 2, 2, name="h_conv1", use_sn=USE_SN))

    # h_conv2 = tf.nn.leaky_relu(ops.conv2d(h_conv1, 64, 5, 5, 2, 2, name="h_conv2", use_sn=USE_SN))

    # h_conv3 = tf.nn.leaky_relu(ops.conv2d(h_conv2, 128, 5, 5, 2, 2, name="h_conv3", use_sn=USE_SN))
    # h_conv3_flat = tf.reshape(h_conv3, [-1, 4*4*128])

    conv1 = tf.nn.relu(ops.conv2d(x, 16, 3, 3, 1, 1, name="conv1", use_sn=USE_SN))

    # 32x32x3 -> 32x32x16
    res1_1 = residual_block(conv1, 16, False, "res1_1")
    res1_2 = residual_block(res1_1, 16, False, "res1_2")
    res1_3 = residual_block(res1_2, 16, False, "res1_3")

    ...

    # 32x32x16 -> 16x16x32
    res2_1 = residual_block(res1_3, 32, True, "res2_1")
    res2_2 = residual_block(res2_1, 32, False, "res2_2")
    res2_3 = residual_block(res2_2, 32, False, "res2_3")

    ...

    # 16x16x32 -> 8x8x64
    res3_1 = residual_block(res2_3, 64, True, "res3_1")
    res3_2 = residual_block(res3_1, 64, False, "res3_2")
    res3_3 = residual_block(res3_2, 64, False, "res3_3")

    ...

    res3_flat = tf.reshape(res3_3, [-1, 8*8*64])

    f1 = tf.nn.leaky_relu(ops.linear(res3_flat, 1024, scope="f1", use_sn=USE_SN))

    if LABEL_BASED_DISCRIMINATOR:
        f2 = ops.linear(f1, 11, scope="f2", use_sn=USE_SN)
        return f2
    else:
        f2 = tf.nn.sigmoid(ops.linear(f1, 1, scope="f2", use_sn=USE_SN))
        return f2

with tf.variable_scope('G'):
    z = tf.random_uniform([100, 100])
    G = generator(z)

image_summary = tf.summary.image("generated image", G, 10)

images_and_labels = load_images_and_labels(100).get_next()
x = images_and_labels[0]
x.set_shape([100, 32, 32, 3])
yx = images_and_labels[1]
x_indices = images_and_labels[2]
yg = tf.reshape(tf.tile(tf.one_hot(10, 11), [100]), [100, 11])

with tf.variable_scope('D'):
    Dx, _, _ = resnet_architecture.resnet_cifar_discriminator(x, True, consts.SPECTRAL_NORM, reuse=False)
    Dg, _, _ = resnet_architecture.resnet_cifar_discriminator(G, True, consts.SPECTRAL_NORM, reuse=True)
#     Dx = discriminator(x)
# with tf.variable_scope('D', reuse=True):
#     Dg = discriminator(G)

if LABEL_BASED_DISCRIMINATOR:
    loss_d = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yx, logits=Dx) + tf.nn.softmax_cross_entropy_with_logits(labels=yg, logits=Dg)) #This optimizes the discriminator.
    loss_g = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yg, logits=Dg)) #This optimizes the generator.
else:
    loss_d = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
    loss_g = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.
loss_d_summary = tf.summary.scalar("discriminator loss", loss_d)
loss_g_summary = tf.summary.scalar("generator loss", loss_g)

if LABEL_BASED_DISCRIMINATOR:
    real_correct_prediction = tf.equal(tf.argmax(Dx, 1), tf.argmax(yx, 1))
    real_accuracy = tf.reduce_mean(tf.cast(real_correct_prediction, tf.float32))

    generated_correct_prediction = tf.equal(tf.argmax(Dg, 1), tf.argmax(yg, 1))
    generated_accuracy = tf.reduce_mean(tf.cast(generated_correct_prediction, tf.float32))

vars = tf.trainable_variables()
for v in vars:
    print(v.name)
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

d_opt = tf.train.AdamOptimizer(1e-4).minimize(loss_d, var_list=d_params)
g_opt = tf.train.AdamOptimizer(1e-4).minimize(loss_g, var_list=g_params)

d_saver = tf.train.Saver(d_params)
g_saver = tf.train.Saver(g_params)

start_time = time.time()
sample_directory = 'generated_images/cifar/{}'.format(start_time)
if not os.path.exists(sample_directory):
    os.makedirs(sample_directory)
shutil.copy(os.path.abspath(__file__), sample_directory)

with tf.Session() as session:
    writer = tf.summary.FileWriter(sample_directory, session.graph)

    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
        print('attempting to load checkpoint from {}'.format(checkpoint_dir))
        
        d_checkpoint_dir = checkpoint_dir + "/disciminator_checkpoints"
        d_checkpoint = tf.train.get_checkpoint_state(d_checkpoint_dir)
        if d_checkpoint and d_checkpoint.model_checkpoint_path:
            d_saver.restore(session, d_checkpoint.model_checkpoint_path)
            print(d_checkpoint)

        g_checkpoint_dir = checkpoint_dir + "/generator_checkpoints"
        g_checkpoint = tf.train.get_checkpoint_state(g_checkpoint_dir)
        if g_checkpoint and g_checkpoint.model_checkpoint_path:
            g_saver.restore(session, g_checkpoint.model_checkpoint_path)
            print(g_checkpoint)
    else:
        print('no checkpoint specified, starting training from scratch')

    previous_step_time = time.time()
    d_epoch_losses = []
    g_epoch_losses = []
    for step in range(1, 200001):
        # update discriminator
        for i in range(5):
            summary, d_batch_loss, _ = session.run([loss_d_summary, loss_d, d_opt])
            d_epoch_losses.append(d_batch_loss)
            writer.add_summary(summary, step)

        # update generator
        for i in range(1):
            summary, g_batch_loss, _ = session.run([loss_g_summary, loss_g, g_opt])
            g_epoch_losses.append(g_batch_loss)
            writer.add_summary(summary, step)

        if step % 100 == 0:
            print('{}: discriminator loss {:.8f}\tgenerator loss {:.8f}'.format(step, np.mean(d_epoch_losses), np.mean(g_epoch_losses)))
            d_epoch_losses = []
            g_epoch_losses = []

        if step % 100 == 0 and LABEL_BASED_DISCRIMINATOR:
            real_train_accuracy, generated_train_accuracy = session.run([real_accuracy, generated_accuracy])
            print('label accuracy: {}'.format(real_train_accuracy))
            print('real/fake accurary: {}'.format(generated_train_accuracy))

        if step % 100 == 0:
            current_step_time = time.time()
            print('{}: previous 100 steps took {:.4f}s'.format(step, current_step_time - previous_step_time))
            previous_step_time = current_step_time

        if step % 1000 == 0:
            summary, gen_image, discriminator_confidence = session.run([image_summary, G, Dg])
            save_images.save_images(np.reshape(gen_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            writer.add_summary(summary, step)
            min_discriminator_confidence = np.min(discriminator_confidence)
            max_discriminator_confidence = np.max(discriminator_confidence)
            print("minimum discriminator confidence: {:.4f} maximum discriminator confidence: {:.4f}\n".format(min_discriminator_confidence, max_discriminator_confidence))
            min_confidence_index = np.argmin(discriminator_confidence)
            max_confidence_index = np.argmax(discriminator_confidence)
            min_max_image = np.ndarray([2, 32, 32, 3])
            min_max_image[0] = gen_image[min_confidence_index]
            min_max_image[1] = gen_image[max_confidence_index]
            print("minimum confidence index: {} maximum confidence index: {}".format(min_confidence_index, max_confidence_index))
            save_images.save_images(min_max_image, [2, 1], sample_directory + '/{}gen_min_max.png'.format(step))

        if step % 1000 == 0 and OUTPUT_REAL_IMAGES:
            real_image, real_labels = session.run([x, x_indices])
            save_images.save_images(np.reshape(real_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}real.png'.format(step))
            print('Real image labels:\n{}'.format([CIFAR_CATEGORIES[rl] for rl in real_labels]))


        if step % 1000 == 0:
            d_checkpoint_dir = sample_directory + "/disciminator_checkpoints"
            if not os.path.exists(d_checkpoint_dir):
                os.makedirs(d_checkpoint_dir)
            d_saver.save(session, d_checkpoint_dir + '/discriminator.model', global_step=step)

            g_checkpoint_dir = sample_directory + "/generator_checkpoints"
            if not os.path.exists(g_checkpoint_dir):
                os.makedirs(g_checkpoint_dir)
            g_saver.save(session, g_checkpoint_dir + '/generator.model', global_step=step)