"""Basic GAN to generate cifar images."""
import tensorflow as tf


import resnet_architecture
import consts
import cifar_models
import cifar_loader

from gan import GAN

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/home/lucas/training_data/cifar10/", "Directory to read the training data from.")

class CIFAR_GAN(GAN):
    def __init__(self, training_steps, batch_size, label_based_discriminator, output_real_images = False):
        super().__init__(32, 32, "cifar", training_steps, batch_size, label_based_discriminator, output_real_images = False)

    def generator(self, z):
        G = cifar_models.resnet_generator(z)
        # G = resnet_architecture.resnet_cifar_generator(z, True)
        return G

    def discriminator(self, x, label_based_discriminator):
        Dx, Dx_logits, _ = cifar_models.resnet_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=label_based_discriminator)
        # Dx, Dx_logits, _ = resnet_architecture.resnet_cifar_discriminator(x, True, consts.SPECTRAL_NORM, reuse=False)
        return Dx, Dx_logits

    def load_data(self, batch_size):
        images_and_labels = cifar_loader.load_images_and_labels(batch_size, FLAGS.data_dir).get_next()
        x = images_and_labels[0]
        x.set_shape([batch_size, 3, 32, 32])
        yx = images_and_labels[1]
        yg = tf.reshape(tf.tile(tf.one_hot(10, 11), [batch_size]), [batch_size, 11])

        return x, yx, yg

g = CIFAR_GAN(40000, 100, True)
g.run()