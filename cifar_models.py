import tensorflow as tf

import ops
import resnet_blocks


def conv_generator(z):
    with tf.variable_scope('generator'):
        f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
        f1 = tf.layers.batch_normalization(f1, training=True)

        f2 = tf.layers.dense(f1, 4*4*128, tf.nn.leaky_relu)
        f2 = tf.reshape(f2, [-1, 4, 4, 128])
        f2 = tf.layers.batch_normalization(f2, training=True)

        conv1 = tf.layers.conv2d_transpose(f2, 32, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.leaky_relu)
        conv1 = tf.layers.batch_normalization(conv1, training=True)

        conv2 = tf.layers.conv2d_transpose(conv1, 32, [5, 5], strides=(2, 2), padding="same", activation=tf.nn.leaky_relu)
        conv2 = tf.layers.batch_normalization(conv2, training=True)

        conv3 = tf.layers.conv2d_transpose(conv2, 3, (3, 3), strides=(2, 2), padding="same", activation=tf.nn.tanh)

        return conv3

def conv_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h_conv1 = tf.nn.leaky_relu(ops.conv2d(x, 32, 5, 5, 2, 2, name="h_conv1", use_sn=use_sn))

        h_conv2 = tf.nn.leaky_relu(ops.conv2d(h_conv1, 64, 5, 5, 2, 2, name="h_conv2", use_sn=use_sn))

        h_conv3 = tf.nn.leaky_relu(ops.conv2d(h_conv2, 128, 5, 5, 2, 2, name="h_conv3", use_sn=use_sn))
        h_conv3_flat = tf.reshape(h_conv3, [-1, 4*4*128])

        if label_based_discriminator:
            f1_logit = ops.linear(h_conv3_flat, 11, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
        else:
            f1_logit = ops.linear(h_conv3_flat, 1, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None

def resnet_generator(z):
    with tf.variable_scope('generator'):
        f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
        f1 = tf.layers.batch_normalization(f1, training=True)

        f2 = tf.layers.dense(f1, 128*4*4, tf.nn.leaky_relu)
        f2 = tf.reshape(f2, [-1, 128, 4, 4])
        f2 = tf.layers.batch_normalization(f2, training=True)

        res1_1 = resnet_blocks.generator_residual_block(f2, 128, True, "res1_1")
        res1_2 = resnet_blocks.generator_residual_block(res1_1, 128, False, "res1_2")

        res2_1 = resnet_blocks.generator_residual_block(res1_2, 128, True, "res2_1")
        res2_2 = resnet_blocks.generator_residual_block(res2_1, 128, False, "res2_2")

        conv = tf.layers.conv2d_transpose(res2_2, 3, (3, 3), strides=(2, 2), padding="same", activation=tf.nn.tanh, data_format="channels_first")

        return conv



def resnet_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 32x32x3 -> 16x16x32
        res1_1 = resnet_blocks.discriminator_residual_block(x, 128, True, "res1_1", use_sn=use_sn, reuse=reuse)
        # res1_2 = resnet_blocks.discriminator_residual_block(res1_1, 128, False, "res1_2", use_sn=use_sn, reuse=reuse)
        # res1_3 = resnet_blocks.discriminator_residual_block(res1_2, 128, False, "res1_3", use_sn=use_sn, reuse=reuse)

        # 16x16x32 -> 8x8x64
        res2_1 = resnet_blocks.discriminator_residual_block(res1_1, 128, True, "res2_1", use_sn=use_sn, reuse=reuse)
        res2_2 = resnet_blocks.discriminator_residual_block(res2_1, 128, False, "res2_2", use_sn=use_sn, reuse=reuse)
        res2_3 = resnet_blocks.discriminator_residual_block(res2_2, 128, False, "res2_3", use_sn=use_sn, reuse=reuse)

        # 8x8x64 -> 4x4x128
        # res3_1 = resnet_blocks.discriminator_residual_block(res2_3, 128, True, "res3_1", use_sn=use_sn, reuse=reuse)
        # res3_2 = resnet_blocks.discriminator_residual_block(res3_1, 128, False, "res3_2", use_sn=use_sn, reuse=reuse)
        # res3_3 = resnet_blocks.discriminator_residual_block(res3_2, 128, False, "res3_3", use_sn=use_sn, reuse=reuse)

        res3_flat = tf.reshape(res2_3, [100, 128*8*8])

        if label_based_discriminator:
            f1_logit = ops.linear(res3_flat, 11, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
        else:
            f1_logit = ops.linear(res3_flat, 1, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
