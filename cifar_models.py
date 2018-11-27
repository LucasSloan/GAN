import tensorflow as tf

import resnet_architecture
import ops

def generator_residual_block(input, channels, upsample):
    shortcut = input
    strides = (1, 1)
    if upsample:
        strides = (2, 2)
        shortcut = tf.layers.conv2d_transpose(shortcut, channels, (1, 1), strides=strides, padding="same")

    conv1 = tf.layers.conv2d_transpose(input, channels, (3, 3), strides=strides, padding="same")
    conv1 = tf.layers.batch_normalization(conv1, training=True)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = tf.layers.conv2d(conv1, channels, (3,3), strides=(1, 1), padding="same")
    conv2 = tf.layers.batch_normalization(conv2, training=True)

    conv2 += shortcut
    output = tf.nn.leaky_relu(conv2)

    return output

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

def resnet_generator(z):
    with tf.variable_scope('generator'):
        f1 = tf.layers.dense(z, 1024, tf.nn.leaky_relu)
        f1 = tf.layers.batch_normalization(f1, training=True)

        f2 = tf.layers.dense(f1, 4*4*128, tf.nn.leaky_relu)
        f2 = tf.reshape(f2, [-1, 4, 4, 128])
        f2 = tf.layers.batch_normalization(f2, training=True)

        res1_1 = generator_residual_block(f2, 32, True)
        res1_2 = generator_residual_block(res1_1, 32, False)

        res2_1 = generator_residual_block(res1_2, 16, True)
        res2_2 = generator_residual_block(res2_1, 16, False)

        conv = tf.layers.conv2d_transpose(res2_2, 3, (3, 3), strides=(2, 2), padding="same", activation=tf.nn.tanh)

        return conv

def discriminator_residual_block(input, channels, downsample, name, use_sn=True):
    shortcut = input
    stride = 1
    if downsample:
        stride = 2
        shortcut = ops.conv2d(shortcut, channels, 1, 1, stride, stride, name=name + "_shortcut", use_sn=use_sn)
        shortcut = tf.layers.batch_normalization(shortcut, training=True)
        
    conv1 = ops.conv2d(input, channels, 3, 3, 1, 1, name=name + "_conv1", use_sn=use_sn)
    conv1 = tf.layers.batch_normalization(conv1, training=True)
    conv1 = tf.nn.leaky_relu(conv1)

    conv2 = ops.conv2d(conv1, channels, 3, 3, stride, stride, name=name + "_conv2", use_sn=use_sn)
    conv2 = tf.layers.batch_normalization(conv2, training=True)


    conv2 += shortcut
    output = tf.nn.leaky_relu(conv2)

    return output

def conv_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h_conv1 = tf.nn.leaky_relu(ops.conv2d(x, 32, 5, 5, 2, 2, name="h_conv1", use_sn=use_sn))

        h_conv2 = tf.nn.leaky_relu(ops.conv2d(h_conv1, 64, 5, 5, 2, 2, name="h_conv2", use_sn=use_sn))

        h_conv3 = tf.nn.leaky_relu(ops.conv2d(h_conv2, 128, 5, 5, 2, 2, name="h_conv3", use_sn=use_sn))
        h_conv3_flat = tf.reshape(h_conv3, [-1, 4*4*128])

        if label_based_discriminator:
            f1 = ops.linear(h_conv3_flat, 11, scope="f1", use_sn=use_sn)
            return f1
        else:
            f1_logit = ops.linear(h_conv3_flat, 1, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None


def resnet_discriminator(x, reuse=False, use_sn=True, label_based_discriminator=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 32x32x3 -> 16x16x16
        res1_1 = discriminator_residual_block(x, 16, True, "res1_1", use_sn=use_sn)
        res1_2 = discriminator_residual_block(res1_1, 16, False, "res1_2", use_sn=use_sn)
        res1_3 = discriminator_residual_block(res1_2, 16, False, "res1_3", use_sn=use_sn)

        # 16x16x16 -> 8x8x32
        res2_1 = discriminator_residual_block(res1_3, 32, True, "res2_1", use_sn=use_sn)
        res2_2 = discriminator_residual_block(res2_1, 32, False, "res2_2", use_sn=use_sn)
        res2_3 = discriminator_residual_block(res2_2, 32, False, "res2_3", use_sn=use_sn)

        # 8x8x32 -> 4x4x64
        res3_1 = discriminator_residual_block(res2_3, 64, True, "res3_1", use_sn=use_sn)
        res3_2 = discriminator_residual_block(res3_1, 64, False, "res3_2", use_sn=use_sn)
        res3_3 = discriminator_residual_block(res3_2, 64, False, "res3_3", use_sn=use_sn)

        res3_flat = tf.reshape(res3_3, [-1, 4*4*64])

        if label_based_discriminator:
            f1 = ops.linear(res3_flat, 11, scope="f1", use_sn=use_sn)
            return f1
        else:
            f1_logit = ops.linear(res3_flat, 1, scope="f1", use_sn=use_sn)
            f1 = tf.nn.sigmoid(f1_logit)
            return f1, f1_logit, None
