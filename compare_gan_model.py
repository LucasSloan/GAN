import time

import numpy as np
import tensorflow as tf

import consts
import resnet_architecture
import cifar_loader
import save_images

IS_TRAINING = True
BATCH_SIZE = 100
Z_DIM = 100
TRAINING_STEPS = 200000
DISC_ITERS = 5

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/home/lucas/training_data/cifar10/",
                    "Directory to read the training data from.")


def get_optimizer(learning_rate, name_prefix, beta1=0.5, beta2=0.999):
    return tf.train.AdamOptimizer(
        learning_rate,
        beta1=beta1,
        beta2=beta2,
        name=name_prefix + "adam")


def discriminator(x, reuse=False, architecture=consts.RESNET_CIFAR, discriminator_normalization=consts.NO_NORMALIZATION):
    if architecture == consts.RESNET5_ARCH:
        return resnet_architecture.resnet5_discriminator(
            x, IS_TRAINING, discriminator_normalization, reuse)
    elif architecture == consts.RESNET107_ARCH:
        return resnet_architecture.resnet107_discriminator(
            x, IS_TRAINING, discriminator_normalization, reuse)
    elif architecture == consts.RESNET_CIFAR:
        return resnet_architecture.resnet_cifar_discriminator(
            x, IS_TRAINING, discriminator_normalization, reuse)
    else:
        raise NotImplementedError(
            "Architecture %s not implemented." % architecture)


def generator(z, reuse=False, architecture=consts.RESNET_CIFAR):
    if architecture == consts.RESNET5_ARCH:
        # TODO: handle RESNET5's variable output shape
        return resnet_architecture.resnet5_generator(
            z,
            is_training=IS_TRAINING,
            reuse=reuse,
            colors=3,
            output_shape=128)
    elif architecture == consts.RESNET107_ARCH:
        return resnet_architecture.resnet107_generator(
            z, is_training=IS_TRAINING, reuse=reuse, colors=3)
    elif architecture == consts.RESNET_CIFAR:
        return resnet_architecture.resnet_cifar_generator(
            z, is_training=IS_TRAINING, reuse=reuse, colors=3)
    else:
        raise NotImplementedError(
            "Architecture %s not implemented." % architecture)


def check_variables(t_vars, d_vars, g_vars):
    """Make sure that every variable belongs to generator or discriminator."""
    shared_vars = set(d_vars) & set(g_vars)
    if shared_vars:
        raise ValueError("Shared trainable variables: %s" % shared_vars)
    unused_vars = set(t_vars) - set(d_vars) - set(g_vars)
    if unused_vars:
        raise ValueError("Unused trainable variables: %s" % unused_vars)


def print_progress(step, start_step, start_time, d_loss, g_loss):
    if step % 100 == 0:
      time_elapsed = time.time() - start_time
      steps_per_sec = (step - start_step) / time_elapsed
      eta_seconds = (TRAINING_STEPS - step) / (steps_per_sec + 0.0000001)
      eta_minutes = eta_seconds / 60.0
      print("[%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f "
            "steps_per_sec: %.4f ETA: %.2f minutes" %
            (step, TRAINING_STEPS, time_elapsed, d_loss, g_loss,
             steps_per_sec, eta_minutes))

def maybe_save_samples(step, gen_image, sample_directory):
    if step % 1000 == 0:
        save_images.save_images(np.reshape(gen_image, [100, 32, 32, 3]), [10, 10], sample_directory + '/{}gen.png'.format(step))


# Input images.
images = cifar_loader.load_images_and_labels(
    BATCH_SIZE, FLAGS.data_dir).get_next()[0]
images.set_shape([BATCH_SIZE, 32, 32, 3])
# Noise vector.
z = tf.random_uniform([BATCH_SIZE, Z_DIM])

# Discriminator output for real images.
d_real, d_real_logits, _ = discriminator(images)

# Discriminator output for fake images.
generated = generator(z)
d_fake, d_fake_logits, _ = discriminator(generated, reuse=True)

# Define the loss functions
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits, labels=tf.ones_like(d_real)))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.ones_like(d_fake)))

# Divide trainable variables into a group for D and group for G.
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if "discriminator" in var.name]
g_vars = [var for var in t_vars if "generator" in var.name]
check_variables(t_vars, d_vars, g_vars)

# Define optimization ops.
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_optim = get_optimizer(1e-4, "d_").minimize(d_loss, var_list=d_vars)
    g_optim = get_optimizer(1e-4, "g_").minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    """Runs the training algorithm."""

    # Initialize the variables.
    global_step = tf.train.get_or_create_global_step()
    global_step_inc = global_step.assign_add(1)
    tf.global_variables_initializer().run()

    # Start training.
    counter = tf.train.global_step(sess, global_step)
    start_time = time.time()
    sample_directory = 'generated_images/cifar/{}'.format(start_time)

    gl = None
    start_step = int(counter) + 1
    for step in range(start_step, TRAINING_STEPS + 1):
        # Update the discriminator network.
        _, dl = sess.run([d_optim, d_loss])

        # Update the generator network.
        if (counter - 1) % DISC_ITERS == 0 or g_loss is None:
            _, gl, gen_image = sess.run([g_optim, g_loss, generated])

        sess.run(global_step_inc)
        print_progress(step, start_step, start_time, dl, gl)
        maybe_save_samples(step, gen_image, sample_directory)
