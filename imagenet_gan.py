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


def resnet_generator(z):
    with tf.variable_scope('generator'):
        f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
        f1 = tf.layers.batch_normalization(f1, training=True)

        f2 = tf.layers.dense(f1, 4*4*128, tf.nn.leaky_relu)
        f2 = tf.reshape(f2, [-1, 4, 4, 128])
        f2 = tf.layers.batch_normalization(f2, training=True)

        res1_1 = resnet_blocks.generator_residual_block(
            f2, 128, True, "res1_1")
        res1_2 = resnet_blocks.generator_residual_block(
            res1_1, 128, True, "res1_2")

        res2_1 = resnet_blocks.generator_residual_block(
            res1_2, 128, True, "res2_1")
        res2_2 = resnet_blocks.generator_residual_block(
            res2_1, 128, False, "res2_2")

        conv = tf.layers.conv2d_transpose(res2_2, 3, (3, 3), strides=(
            2, 2), padding="same", activation=tf.nn.tanh)

        return conv


def resnet_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 32x32x3 -> 16x16x32
        res1_1 = resnet_blocks.discriminator_residual_block(
            x, 128, True, "res1_1", use_sn=use_sn, reuse=reuse)
        # res1_2 = resnet_blocks.discriminator_residual_block(res1_1, 128, False, "res1_2", use_sn=use_sn, reuse=reuse)
        # res1_3 = resnet_blocks.discriminator_residual_block(res1_2, 128, False, "res1_3", use_sn=use_sn, reuse=reuse)

        # 16x16x32 -> 8x8x64
        res2_1 = resnet_blocks.discriminator_residual_block(
            res1_1, 128, True, "res2_1", use_sn=use_sn, reuse=reuse)
        res2_2 = resnet_blocks.discriminator_residual_block(
            res2_1, 128, True, "res2_2", use_sn=use_sn, reuse=reuse)
        res2_3 = resnet_blocks.discriminator_residual_block(
            res2_2, 128, False, "res2_3", use_sn=use_sn, reuse=reuse)

        # 8x8x64 -> 4x4x128
        # res3_1 = resnet_blocks.discriminator_residual_block(res2_3, 128, True, "res3_1", use_sn=use_sn, reuse=reuse)
        # res3_2 = resnet_blocks.discriminator_residual_block(res3_1, 128, False, "res3_2", use_sn=use_sn, reuse=reuse)
        # res3_3 = resnet_blocks.discriminator_residual_block(res3_2, 128, False, "res3_3", use_sn=use_sn, reuse=reuse)

        res3_flat = tf.reshape(res2_3, [-1, 8*8*128])

        if label_based_discriminator:
            f1_logit = ops.linear(res3_flat, 1001, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
        else:
            f1_logit = ops.linear(res3_flat, 1, scope="f1", use_sn=use_sn)
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
