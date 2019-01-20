import tensorflow as tf

import ops
import resnet_blocks
import non_local


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

G_DIM = 64
def resnet_generator(z, labels):
    with tf.variable_scope('generator'):
        noise_plus_labels = tf.concat([z, labels], 1)
        linear = ops.linear(noise_plus_labels, G_DIM * 4 * 4 * 4, use_sn=True)
        linear = tf.reshape(linear, [-1, G_DIM * 4, 4, 4])

        res1 = resnet_blocks.generator_residual_block(
            linear, G_DIM * 4, True, "res1") # 8x8
        res2 = resnet_blocks.generator_residual_block(
            res1, G_DIM * 2, True, "res2") # 16x16
        nl = non_local.sn_non_local_block_sim(res2, None, name='nl')
        res3 = resnet_blocks.generator_residual_block(
            nl, G_DIM, True, "res3") # 32x32
        res3 = tf.layers.batch_normalization(res3, training=True)

        conv = ops.conv2d(res3, 3, 3, 3, 1, 1, name = "conv", use_sn=True)
        conv = tf.nn.tanh(conv)

        return conv



D_DIM = 64
def resnet_discriminator(x, labels, reuse=False, use_sn=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        res1 = resnet_blocks.discriminator_residual_block(
            x, D_DIM, True, "res1", use_sn=use_sn, reuse=reuse) # 16x16
        nl = non_local.sn_non_local_block_sim(res1, None, name="nl")
        res2 = resnet_blocks.discriminator_residual_block(
            nl, D_DIM * 2, True, "res2", use_sn=use_sn, reuse=reuse) # 8x8
        res3 = resnet_blocks.discriminator_residual_block(
            res2, D_DIM * 4, True, "res3", use_sn=use_sn, reuse=reuse) # 4x4
        res4 = resnet_blocks.discriminator_residual_block(
            res3, D_DIM * 4, False, "res4", use_sn=use_sn, reuse=reuse) # 4x4

        res4_flat = tf.reshape(res4, [-1, 4 * 4 * D_DIM * 4])

        flat_plus_labels = tf.concat([res4_flat, labels], 1)

        f1_logit = ops.linear(flat_plus_labels, 1, scope="f1", use_sn=use_sn)
        f1 = tf.nn.sigmoid(f1_logit)
        return f1, f1_logit, None
