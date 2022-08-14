"""Basic GAN to generate cifar images."""
import tensorflow as tf


import resnet_architecture
import consts
import cifar_models
import cifar_loader

from gan import GAN

tf.compat.v1.disable_eager_execution()

class CIFAR_GAN(GAN):
    def __init__(self, training_steps, batch_size, output_real_images = False):
        super().__init__(32, 32, "cifar", training_steps, batch_size, 10, output_real_images = output_real_images)

    def generator(self, z, labels):
        G = cifar_models.resnet_generator(z, labels)
        # G = resnet_architecture.resnet_cifar_generator(z, True)
        return G

    def discriminator(self, x, labels):
        Dx, Dx_logits, _ = cifar_models.resnet_discriminator(x, labels, reuse=False, use_sn=True)
        # Dx, Dx_logits, _ = resnet_architecture.resnet_cifar_discriminator(x, True, consts.SPECTRAL_NORM, reuse=False)
        return Dx, Dx_logits

    def load_data(self, batch_size):
        images_and_labels = cifar_loader.load_images_and_labels(batch_size).get_next()
        x = images_and_labels[0]
        x.set_shape([batch_size, 3, 32, 32])
        yx = images_and_labels[2]
        labels = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
        yg = tf.one_hot(labels, 10)

        return x, yx, yg

g = CIFAR_GAN(200000, 200, True)
g.run()