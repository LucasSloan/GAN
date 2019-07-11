import tensorflow as tf

import loader
import os


def parse_images(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_flipped = tf.image.random_flip_left_right(image_decoded)
    image_hue = tf.image.random_hue(image_flipped, 0.08)
    image_sat = tf.image.random_saturation(image_hue, 0.6, 1.6)
    image_bright = tf.image.random_brightness(image_sat, 0.05)
    image_contrast = tf.image.random_contrast(image_bright, 0.7, 1.3)

    image_normalized = 2.0 * \
        tf.image.convert_image_dtype(image_contrast, tf.float32) - 1.0
    image_nchw = tf.transpose(image_normalized, [2, 0, 1])

    filename = tf.reshape(filename, [1])
    path_parts = tf.string_split(filename, os.sep)
    dir = path_parts.values[-2]
    int_label = loader.text_to_index(dir)
    one_hot = loader.text_to_one_hot(dir, 9)

    return image_nchw, one_hot, int_label


def load_images_and_labels(batch_size):
    image_files_dataset = tf.data.Dataset.list_files(
        "D:\\cifar10\\32x32\\*\\*")
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=8)

    dataset = image_dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()
