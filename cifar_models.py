import tensorflow as tf

import ops
import resnet_blocks
import non_local


def conv_generator(z):
    with tf.compat.v1.variable_scope('generator'):
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
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
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
def simple_resnet_generator(z, labels):
    with tf.compat.v1.variable_scope('generator'):
        linear = tf.layers.dense(z, G_DIM * 4 * 4 * 4)
        linear = tf.reshape(linear, [-1, G_DIM * 4, 4, 4])

        res1 = resnet_blocks.simple_generator_block(linear, G_DIM * 4, "res1") # 8x8
        res2 = resnet_blocks.simple_generator_block(res1, G_DIM * 2, "res2") # 16x16
        res3 = resnet_blocks.simple_generator_block(res2, G_DIM, "res3") # 32x32
        res3 = tf.layers.batch_normalization(res3, training=True)
        print(res3.shape)
        res3 = tf.nn.relu(res3)

        conv = tf.layers.conv2d(res3, 3, (3, 3), padding="same", data_format="channels_first")
        conv = tf.nn.tanh(conv)

        return conv



D_DIM = 64
def simple_resnet_discriminator(x, labels, reuse=False, use_sn=True):
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
        res1 = resnet_blocks.simple_discriminator_block(x, D_DIM, "res1") # 16x16
        res2 = resnet_blocks.simple_discriminator_block(res1, D_DIM * 2, "res2") # 8x8
        res3 = resnet_blocks.simple_discriminator_block(res2, D_DIM * 4, "res3") # 4x4

        res3_flat = tf.reshape(res3, [-1, 4*4*D_DIM * 4])

        f1_logit = tf.layers.dense(res3_flat, 1)
        f1 = tf.nn.sigmoid(f1_logit)
        return f1, f1_logit, None

G_DIM = 64
def resnet_generator(z, labels):
    with tf.compat.v1.variable_scope('generator'):
        embedding_map = tf.compat.v1.get_variable(
            name='embedding_map',
            shape=[10, 100],
            initializer=tf.keras.initializers.glorot_normal)
        label_embedding = tf.nn.embedding_lookup(embedding_map, labels)
        noise_plus_labels = tf.concat([z, label_embedding], 1)
        linear = ops.linear(noise_plus_labels, G_DIM * 4 * 4 * 4, use_sn=True)
        linear = tf.reshape(linear, [-1, G_DIM * 4, 4, 4])

        res1 = resnet_blocks.class_conditional_generator_block(
            linear, labels, G_DIM * 4, 10, True, "res1") # 8x8
        res2 = resnet_blocks.class_conditional_generator_block(
            res1, labels, G_DIM * 2, 10, True, "res2") # 16x16
        nl = non_local.sn_non_local_block_sim(res2, None, name='nl')
        res3 = resnet_blocks.class_conditional_generator_block(
            nl, labels, G_DIM, 10, True, "res3") # 32x32
        res3 = tf.compat.v1.layers.batch_normalization(res3, training=True)
        res3 = tf.nn.relu(res3)

        conv = ops.conv2d(res3, 3, 3, 3, 1, 1, name = "conv", use_sn=True)
        conv = tf.nn.tanh(conv)

        return conv



D_DIM = 64
def resnet_discriminator(x, labels, reuse=False, use_sn=True):
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
        res1 = resnet_blocks.discriminator_residual_block(
            x, D_DIM, True, "res1", use_sn=use_sn, reuse=reuse) # 16x16
        nl = non_local.sn_non_local_block_sim(res1, None, name="nl")
        res2 = resnet_blocks.discriminator_residual_block(
            nl, D_DIM * 2, True, "res2", use_sn=use_sn, reuse=reuse) # 8x8
        res3 = resnet_blocks.discriminator_residual_block(
            res2, D_DIM * 4, True, "res3", use_sn=use_sn, reuse=reuse) # 4x4
        res4 = resnet_blocks.discriminator_residual_block(
            res3, D_DIM * 4, False, "res4", use_sn=use_sn, reuse=reuse) # 4x4

        res4 = tf.nn.relu(res4)
        res4_chanels = tf.reduce_sum(res4, [2, 3])
        f1_logit = ops.linear(res4_chanels, 1, scope="f1", use_sn=use_sn)

        embedding_map = tf.compat.v1.get_variable(
            name='embedding_map',
            shape=[10, D_DIM * 4],
            initializer=tf.keras.initializers.glorot_normal)

        label_embedding = tf.nn.embedding_lookup(embedding_map, labels)
        f1_logit += tf.reduce_sum(res4_chanels * label_embedding, axis=1, keepdims=True)

        f1 = tf.nn.sigmoid(f1_logit)
        return f1, f1_logit, None
