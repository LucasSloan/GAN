"""Basic GAN to generate mnist images."""
from gan import GAN
import tensorflow as tf
import cifar_models
import loader
import os

def parse_images(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_image_with_crop_or_pad(
        image_decoded, 32, 32)
    image_normalized = 2.0 * \
        tf.image.convert_image_dtype(image_resized, tf.float32) - 1.0
    image_nchw = tf.transpose(image_normalized, [2, 0, 1])

    filename = tf.reshape(filename, [1])
    path_parts = tf.string_split(filename, os.sep)
    dir = path_parts.values[-2]
    int_label = loader.text_to_index(dir)
    one_hot = loader.text_to_one_hot(dir, 9)

    return image_nchw, one_hot, int_label


def load_images(batch_size):
    image_files_dataset = tf.data.Dataset.list_files("E:\\mnist\\train\\*\\*")
    image_files_dataset = image_files_dataset.concatenate(
        tf.data.Dataset.list_files("E:\\mnist\\test\\*\\*"))
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    dataset = image_dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()


class MNIST_GAN(GAN):
    def __init__(self, training_steps, batch_size, output_real_images = False):
        super().__init__(32, 32, "mnist", training_steps, batch_size, 10, output_real_images = output_real_images)

    def generator(self, z, labels):
        G = cifar_models.resnet_generator(z, labels)
        return G

    def discriminator(self, x, labels):
        Dx, Dx_logits, _ = cifar_models.resnet_discriminator(x, labels, reuse=False, use_sn=True)
        return Dx, Dx_logits

    def load_data(self, batch_size):
        images_and_labels = load_images(batch_size).get_next()
        x = images_and_labels[0]
        x.set_shape([batch_size, 3, 32, 32])
        yx = images_and_labels[1]
        labels = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
        yg = tf.one_hot(labels, 10)

        return x, yx, yg

g = MNIST_GAN(100000, 100, True)
g.run()
