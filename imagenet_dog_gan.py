import tensorflow as tf

from gan import GAN
import resnet_blocks
import ops
import loader
import os
import non_local


image_feature_description = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
}


def parse_images(tfrecord):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    image_decoded = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image_normalized = 2.0 * \
        tf.image.convert_image_dtype(image_decoded, tf.float32) - 1.0
    image_flipped = tf.image.random_flip_left_right(image_normalized)
    image_nchw = tf.transpose(image_flipped, [2, 0, 1])

    raw_label = proto['label'] - 152
    one_hot_label = tf.one_hot(raw_label, 119)

    return image_nchw, one_hot_label, raw_label


def load_imagenet(batch_size):
    files = tf.data.Dataset.list_files("D:\\imagenet\\tfrecords\\64x64_dog\\*")
    raw_dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=16, sloppy=True))
    image_dataset = raw_dataset.map(parse_images, num_parallel_calls=16)

    # dataset = dataset.shuffle(20000)
    # dataset = image_dataset.repeat()
    dataset = image_dataset.apply(tf.contrib.data.shuffle_and_repeat(400000))
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()

G_DIM = 64
def resnet_generator(z, labels):
    with tf.variable_scope('generator'):
        embedding_map = tf.get_variable(
            name='embedding_map',
            shape=[1000, 100],
            initializer=tf.contrib.layers.xavier_initializer())
        label_embedding = tf.nn.embedding_lookup(embedding_map, labels)
        noise_plus_labels = tf.concat([z, label_embedding], 1)
        linear = ops.linear(noise_plus_labels, G_DIM * 8 * 4 * 4, use_sn=True)
        linear = tf.reshape(linear, [-1, G_DIM * 8, 4, 4])

        res1 = resnet_blocks.class_conditional_generator_block(
            linear, labels, G_DIM * 8, 1000, True, "res1") # 8x8
        res2 = resnet_blocks.class_conditional_generator_block(
            res1, labels, G_DIM * 4, 1000, True, "res2") # 16x16
        nl = non_local.sn_non_local_block_sim(res2, None, name='nl')
        res3 = resnet_blocks.class_conditional_generator_block(
            nl, labels, G_DIM * 2, 1000, True, "res3") # 32x32
        res4 = resnet_blocks.class_conditional_generator_block(
            res3, labels, G_DIM, 1000, True, "res4") # 64x64
        res4 = tf.layers.batch_normalization(res4, training=True)
        res4 = tf.nn.relu(res4)

        conv = ops.conv2d(res4, 3, 3, 3, 1, 1, name = "conv", use_sn=True)
        conv = tf.nn.tanh(conv)

        return conv


        # linear = ops.linear(z, G_DIM * 8 * 4 * 4, use_sn=True)
        # linear = tf.reshape(linear, [-1, G_DIM * 8, 4, 4])

        # res1 = resnet_blocks.generator_residual_block(
        #     linear, G_DIM * 8, True, "res1") # 8x8
        # res2 = resnet_blocks.generator_residual_block(
        #     res1, G_DIM * 4, True, "res2") # 16x16
        # nl = non_local.sn_non_local_block_sim(res2, None, name='nl')
        # res3 = resnet_blocks.generator_residual_block(
        #     nl, G_DIM * 2, True, "res3") # 32x32
        # res4 = resnet_blocks.generator_residual_block(
        #     res3, G_DIM, True, "res4") # 64x64
        # res4 = tf.layers.batch_normalization(res4, training=True)

        # conv = ops.conv2d(res4, 3, 3, 3, 1, 1, name = "conv", use_sn=True)
        # conv = tf.nn.tanh(conv)

        # return conv


D_DIM = 64
def resnet_discriminator(x, labels, reuse=False, use_sn=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        res1 = resnet_blocks.discriminator_residual_block(
            x, D_DIM, True, "res1", use_sn=use_sn, reuse=reuse) # 32x32
        res2 = resnet_blocks.discriminator_residual_block(
            res1, D_DIM * 2, True, "res2", use_sn=use_sn, reuse=reuse) # 16x16
        nl = non_local.sn_non_local_block_sim(res2, None, name="nl")
        res3 = resnet_blocks.discriminator_residual_block(
            nl, D_DIM * 4, True, "res3", use_sn=use_sn, reuse=reuse) # 8x8
        res4 = resnet_blocks.discriminator_residual_block(
            res3, D_DIM * 8, True, "res4", use_sn=use_sn, reuse=reuse) # 4x4
        res5 = resnet_blocks.discriminator_residual_block(
            res4, D_DIM * 8, False, "res5", use_sn=use_sn, reuse=reuse) # 4x4

        res5 = tf.nn.relu(res5)
        res5_chanels = tf.reduce_sum(res5, [2, 3])
        f1_logit = ops.linear(res5_chanels, 1, scope="f1", use_sn=use_sn)

        embedding_map = tf.get_variable(
            name='embedding_map',
            shape=[1000, D_DIM * 8],
            initializer=tf.contrib.layers.xavier_initializer())

        label_embedding = tf.nn.embedding_lookup(embedding_map, labels)
        f1_logit += tf.reduce_sum(res5_chanels * label_embedding, axis=1, keepdims=True)

        f1 = tf.nn.sigmoid(f1_logit)
        return f1, f1_logit, None


        # res1 = resnet_blocks.discriminator_residual_block(
        #     x, D_DIM, True, "res1", use_sn=use_sn, reuse=reuse) # 32x32
        # res2 = resnet_blocks.discriminator_residual_block(
        #     res1, D_DIM * 2, True, "res2", use_sn=use_sn, reuse=reuse) # 16x16
        # nl = non_local.sn_non_local_block_sim(res2, None, name="nl")
        # res3 = resnet_blocks.discriminator_residual_block(
        #     nl, D_DIM * 4, True, "res3", use_sn=use_sn, reuse=reuse) # 8x8
        # res4 = resnet_blocks.discriminator_residual_block(
        #     res3, D_DIM * 8, True, "res4", use_sn=use_sn, reuse=reuse) # 4x4
        # res5 = resnet_blocks.discriminator_residual_block(
        #     res4, D_DIM * 8, False, "res5", use_sn=use_sn, reuse=reuse) # 4x4

        # res5_flat = tf.reshape(res5, [-1, G_DIM * 8 * 4 * 4])

        # if label_based_discriminator:
        #     f1_logit = ops.linear(res5_flat, 101, scope="f1", use_sn=use_sn)
        #     f1 = tf.nn.sigmoid(f1_logit)
        #     return f1, f1_logit, None
        # else:
        #     f1_logit = ops.linear(res5_flat, 1, scope="f1", use_sn=use_sn)
        #     f1 = tf.nn.sigmoid(f1_logit)
        #     return f1, f1_logit, None


class IMAGENET_DOG_GAN(GAN):
    def __init__(self, training_steps, batch_size, label_based_discriminator, output_real_images=False):
        super().__init__(64, 64, "imagenet_dog", training_steps, batch_size, 118, output_real_images=True)

    def generator(self, z, labels):
        return resnet_generator(z, labels)

    def discriminator(self, x, labels):
        Dx, Dx_logits, _ = resnet_discriminator(x, labels, reuse=False, use_sn=True)
        return Dx, Dx_logits

    def load_data(self, batch_size):
        images_and_labels = load_imagenet(batch_size).get_next()
        x = images_and_labels[0]
        x.set_shape([batch_size, 3, 64, 64])
        yx = images_and_labels[2]
        yg = tf.reshape(tf.tile(tf.one_hot(118, 119), [batch_size]), [batch_size, 119])

        return x, yx, yg

g = IMAGENET_DOG_GAN(300000, 110, True)
g.run()
