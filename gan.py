"""Base class for GANs"""
import tensorflow as tf
import numpy as np

import time
import os
import shutil
import abc

import save_images


class GAN(abc.ABC):
    def __init__(self, x, y, name, training_steps, batch_size, categories, output_real_images=False):
        self.x = x
        self.y = y
        self.name = name
        self.training_steps = training_steps
        self.batch_size = batch_size
        self.output_real_images = output_real_images
        self.categories = categories

    @abc.abstractmethod
    def generator(self, z, labels):
        pass

    @abc.abstractmethod
    def discriminator(self, x, labels):
        pass

    @abc.abstractmethod
    def load_data(self, batch_size):
        pass

    def losses(self, Dx_logits, Dg_logits, Dx, Dg):
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

        return loss_d, loss_g

    def run(self):
        x, yx, _ = self.load_data(self.batch_size)

        labels = tf.placeholder(tf.int32, [self.batch_size])

        z = 2 * tf.random_uniform([self.batch_size, 100]) - 1

        with tf.variable_scope('D'):
            Dx, Dx_logits = self.discriminator(x, yx)
        with tf.variable_scope('G'):
            G = self.generator(z, labels)
        with tf.variable_scope('D', reuse=True):
            Dg, Dg_logits = self.discriminator(G, labels)

        loss_d, loss_g = self.losses(Dx_logits, Dg_logits, Dx, Dg)

        vars = tf.trainable_variables()
        for v in vars:
            print(v.name)
        d_params = [v for v in vars if v.name.startswith('D/')]
        g_params = [v for v in vars if v.name.startswith('G/')]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_opt = tf.train.AdamOptimizer(
                4e-4, beta1=0.5, beta2=0.999).minimize(loss_d, var_list=d_params)
            g_opt = tf.train.AdamOptimizer(
                1e-4, beta1=0.5, beta2=0.999).minimize(loss_g, var_list=g_params)

        start_time = time.time()
        sample_directory = 'generated_images/{}/{}'.format(
            self.name, start_time)
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        shutil.copy(os.path.abspath(__file__), sample_directory)

        with tf.Session() as session:
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            previous_step_time = time.time()
            d_epoch_losses = []
            g_epoch_losses = []
            for step in range(1, self.training_steps + 1):
                # update discriminator
                gen_labels = np.random.randint(0, self.categories, [self.batch_size])
                d_batch_loss, _ = session.run([loss_d, d_opt], {labels: gen_labels})
                d_epoch_losses.append(d_batch_loss)

                # update generator
                gen_labels = np.random.randint(0, self.categories, [self.batch_size])
                g_batch_loss, _ = session.run([loss_g, g_opt], {labels: gen_labels})
                g_epoch_losses.append(g_batch_loss)

                if step % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    steps_per_sec = 100 / time_elapsed
                    eta_seconds = (self.training_steps - step) / \
                        (steps_per_sec + 0.0000001)
                    eta_minutes = eta_seconds / 60.0
                    print('[{:d}/{:d}] time: {:.2f}s, d_loss: {:.8f}, g_loss: {:.8f}, eta: {:.2f}m'
                          .format(step, self.training_steps, time_elapsed,
                                  np.mean(d_epoch_losses), np.mean(g_epoch_losses), eta_minutes))
                    d_epoch_losses = []
                    g_epoch_losses = []
                    previous_step_time = current_step_time

                if step % 1000 == 0:
                    gen_labels = np.repeat(np.arange(0, self.categories, self.categories / 10), 10)
                    print(gen_labels)
                    gen_image, discriminator_confidence = session.run([G, Dg], {labels: gen_labels})
                    gen_image = np.transpose(gen_image, [0, 2, 3, 1])
                    save_images.save_images(np.reshape(gen_image, [self.batch_size, self.x, self.y, 3]), [
                                            10, 10], sample_directory + '/{}gen.png'.format(step))
                    # min_discriminator_confidence = np.min(discriminator_confidence)
                    # max_discriminator_confidence = np.max(discriminator_confidence)
                    # print("minimum discriminator confidence: {:.4f} maximum discriminator confidence: {:.4f}\n".format(min_discriminator_confidence, max_discriminator_confidence))
                    # min_confidence_index = np.argmin(discriminator_confidence)
                    # max_confidence_index = np.argmax(discriminator_confidence)
                    # min_max_image = np.ndarray([2, self.x, self.y, 3])
                    # min_max_image[0] = gen_image[min_confidence_index]
                    # min_max_image[1] = gen_image[max_confidence_index]
                    # print("minimum confidence index: {} maximum confidence index: {}".format(min_confidence_index, max_confidence_index))
                    # save_images.save_images(min_max_image, [2, 1], sample_directory + '/{}gen_min_max.png'.format(step))

                if step % 1000 == 0 and self.output_real_images:
                    real_image, real_labels = session.run([x, yx])
                    real_image = np.transpose(real_image, [0, 2, 3, 1])
                    save_images.save_images(np.reshape(real_image, [self.batch_size, self.x, self.y, 3]), [
                                            10, 10], sample_directory + '/{}real.png'.format(step))
                    print(real_labels)

            for i in range(self.categories):
                gen_labels = np.tile(i, (100))
                print(gen_labels)
                gen_image, discriminator_confidence = session.run([G, Dg], {labels: gen_labels})
                gen_image = np.transpose(gen_image, [0, 2, 3, 1])
                save_images.save_images(np.reshape(gen_image, [self.batch_size, self.x, self.y, 3]), [
                                        10, 10], sample_directory + '/{}category.png'.format(i))

        total_time = time.time() - start_time
        print("{} steps took {} minutes".format(
            self.training_steps, total_time/60))
