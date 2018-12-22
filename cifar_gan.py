"""Basic GAN to generate cifar images."""
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
import cifar_models
import cifar_loader

# Constants
CIFAR_CATEGORIES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
USE_SN = True
LABEL_BASED_DISCRIMINATOR = True
OUTPUT_REAL_IMAGES = False
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/home/lucas/training_data/cifar10/", "Directory to read the training data from.")

images_and_labels = cifar_loader.load_images_and_labels(100, FLAGS.data_dir).get_next()
x = images_and_labels[0]
x.set_shape([100, 32, 32, 3])
yx = images_and_labels[1]
x_indices = images_and_labels[2]
yg = tf.reshape(tf.tile(tf.one_hot(10, 11), [100]), [100, 11])

z = 2 * tf.random_uniform([100, 100]) - 1

with tf.variable_scope('D'):
    Dx, Dx_logits, _ = cifar_models.resnet_discriminator(x, reuse=False, use_sn=USE_SN, label_based_discriminator=LABEL_BASED_DISCRIMINATOR)
    # Dx, Dx_logits, _ = resnet_architecture.resnet_cifar_discriminator(x, True, consts.SPECTRAL_NORM, reuse=False)

with tf.variable_scope('G'):
    G = cifar_models.resnet_generator(z)
    # G = resnet_architecture.resnet_cifar_generator(z, True)

with tf.variable_scope('D'):
    Dg, Dg_logits, _ = cifar_models.resnet_discriminator(G,  reuse=True, use_sn=USE_SN, label_based_discriminator=LABEL_BASED_DISCRIMINATOR)
    # Dg, Dg_logits, _ = resnet_architecture.resnet_cifar_discriminator(G, True, consts.SPECTRAL_NORM, reuse=True)

if LABEL_BASED_DISCRIMINATOR:
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dx_logits, labels=yx))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg_logits, labels=yg))
    loss_d = d_loss_real + d_loss_fake
    loss_g = -tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg_logits, labels=yg))
else:
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dx_logits, labels=tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg_logits, labels=tf.zeros_like(Dg)))
    loss_d = d_loss_real + d_loss_fake
    loss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg_logits, labels=tf.ones_like(Dg)))

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

# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
d_opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.999).minimize(loss_d, var_list=d_params)
g_opt = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.999).minimize(loss_g, var_list=g_params)

start_time = time.time()
sample_directory = 'generated_images/cifar/{}'.format(start_time)
if not os.path.exists(sample_directory):
    os.makedirs(sample_directory)
shutil.copy(os.path.abspath(__file__), sample_directory)

with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    previous_step_time = time.time()
    d_epoch_losses = []
    g_epoch_losses = []
    for step in range(1, 200001):
        # update discriminator
        d_batch_loss, _ = session.run([loss_d, d_opt])
        d_epoch_losses.append(d_batch_loss)

        # update generator
        if (step - 1) % 5 == 0:
            g_batch_loss, _ = session.run([loss_g, g_opt])
            g_epoch_losses.append(g_batch_loss)

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
            gen_image, discriminator_confidence = session.run([G, Dg])
            save_images.save_images(np.reshape(gen_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}gen.png'.format(step))
            # min_discriminator_confidence = np.min(discriminator_confidence)
            # max_discriminator_confidence = np.max(discriminator_confidence)
            # print("minimum discriminator confidence: {:.4f} maximum discriminator confidence: {:.4f}\n".format(min_discriminator_confidence, max_discriminator_confidence))
            # min_confidence_index = np.argmin(discriminator_confidence)
            # max_confidence_index = np.argmax(discriminator_confidence)
            # min_max_image = np.ndarray([2, 32, 32, 3])
            # min_max_image[0] = gen_image[min_confidence_index]
            # min_max_image[1] = gen_image[max_confidence_index]
            # print("minimum confidence index: {} maximum confidence index: {}".format(min_confidence_index, max_confidence_index))
            # save_images.save_images(min_max_image, [2, 1], sample_directory + '/{}gen_min_max.png'.format(step))

        if step % 1000 == 0 and OUTPUT_REAL_IMAGES:
            real_image, real_labels = session.run([x, x_indices])
            save_images.save_images(np.reshape(real_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}real.png'.format(step))
            print('Real image labels:\n{}'.format([CIFAR_CATEGORIES[rl] for rl in real_labels]))
