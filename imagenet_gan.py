import tensorflow as tf

from gan import GAN
import resnet_blocks
import ops
import loader


def parse_images(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_normalized = 2.0 * \
        tf.image.convert_image_dtype(image_decoded, tf.float32) - 1.0
    image_resized = tf.image.resize_images(image_normalized, [64, 64])
    image_flipped = tf.image.flip_left_right(image_resized)

    filename = tf.reshape(filename, [1])
    path_parts = tf.string_split(filename, "/")
    dir = path_parts.values[-2]
    int_label = loader.text_to_index(dir)
    one_hot = loader.text_to_one_hot(dir, 1000)

    return image_flipped, one_hot, int_label


def load_imagenet(batch_size):
    image_files_dataset = tf.data.Dataset.list_files(
        "/home/lucas/training_data/imagenet/2012_train/*/*")
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=4)

    dataset = image_dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()

G_DIM = 64
def resnet_generator(z):
    with tf.variable_scope('generator'):
        linear = ops.linear(z, 4 * 4 * G_DIM * 8, use_sn=True)
        linear = tf.reshape(linear, [-1, 4, 4, G_DIM * 8])

        res1 = resnet_blocks.generator_residual_block(
            linear, G_DIM * 8, True, "res1") # 8x8
        res2 = resnet_blocks.generator_residual_block(
            res1, G_DIM * 4, True, "res2") # 16x16

        res3 = resnet_blocks.generator_residual_block(
            res2, G_DIM * 2, True, "res3") # 32x32
        res4 = resnet_blocks.generator_residual_block(
            res3, G_DIM, True, "res4") # 64x64
        res4 = tf.layers.batch_normalization(res4, training=True)

        conv = ops.conv2d(res4, 3, 3, 3, 1, 1, name = "conv", use_sn=True)
        conv = tf.nn.tanh(conv)

        return conv


D_DIM = 64
def resnet_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        res1 = resnet_blocks.discriminator_residual_block(
            x, D_DIM, True, "res1", use_sn=use_sn, reuse=reuse) # 32x32
        res2 = resnet_blocks.discriminator_residual_block(
            res1, D_DIM * 2, True, "res2", use_sn=use_sn, reuse=reuse) # 16x16
        res3 = resnet_blocks.discriminator_residual_block(
            res2, D_DIM * 4, True, "res3", use_sn=use_sn, reuse=reuse) # 8x8
        res4 = resnet_blocks.discriminator_residual_block(
            res3, D_DIM * 8, True, "res4", use_sn=use_sn, reuse=reuse) # 4x4
        res5 = resnet_blocks.discriminator_residual_block(
            res4, D_DIM * 8, False, "res5", use_sn=use_sn, reuse=reuse) # 4x4

        res5_flat = tf.reshape(res5, [-1, 4 * 4 * D_DIM * 8])

        if label_based_discriminator:
            f1_logit = ops.linear(res5_flat, 1001, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
        else:
            f1_logit = ops.linear(res5_flat, 1, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None


class IMAGENET_GAN(GAN):
    def __init__(self, training_steps, batch_size, label_based_discriminator, output_real_images=False):
        super().__init__(64, 64, "imagenet", training_steps, batch_size,
                         label_based_discriminator, output_real_images=False)

    def generator(self, z):
        return resnet_generator(z)

    def discriminator(self, x, label_based_discriminator):
        Dx, Dx_logits, _ = resnet_discriminator(
            x, reuse=False, use_sn=True, label_based_discriminator=label_based_discriminator)
        return Dx, Dx_logits

    def load_data(self, batch_size):
        images_and_labels = load_imagenet(batch_size).get_next()
        x = images_and_labels[0]
        x.set_shape([batch_size, 64, 64, 3])
        yx = images_and_labels[1]
        yg = tf.reshape(tf.tile(tf.one_hot(1000, 1001), [batch_size]), [batch_size, 1001])

        return x, yx, yg

g = IMAGENET_GAN(100000, 100, True)
g.run()
