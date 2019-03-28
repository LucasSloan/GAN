import resnet_architecture
import consts
import tensorflow as tf
import ops

# def generator_residual_block(input, channels, upsample):
# shortcut = input
# if upsample:
#     shortcut = resnet_architecture.get_conv(shortcut, 0, channels, "up", name + "_shortcut", False)

# conv1 = resnet_architecture.get_conv(input, 0, channels, "up", name + "_conv1", False)
# conv1 = tf.layers.batch_normalization(conv1, training=True)
# conv1 = tf.nn.leaky_relu(conv1)

# conv2 = resnet_architecture.get_conv(shortcut, 0, channels, "none", name + "_conv2", False)
# conv2 = tf.layers.batch_normalization(conv2, training=True)

# conv2 += shortcut
# output = tf.nn.leaky_relu(conv2)

# return output


def generator_residual_block(input, channels, upsample, name):
    scale = "none"
    if upsample:
        scale = "up"
    return resnet_architecture.generator_block(input, in_channels=0,
                                               out_channels=channels,
                                               scale=scale, block_scope=name,
                                               is_training=True, reuse=False)


def discriminator_residual_block(input, channels, downsample, name, use_sn=True, reuse=False):
    scale = "none"
    if downsample:
        scale = "down"
    discriminator_normalization = consts.NO_NORMALIZATION
    if use_sn:
        discriminator_normalization = consts.SPECTRAL_NORM
    return resnet_architecture.discriminator_block(
        input, in_channels=0, out_channels=channels,
        scale=scale, block_scope=name,
        is_training=True, reuse=reuse,
        discriminator_normalization=discriminator_normalization)

    # shortcut = input
    # stride = 1
    # if downsample:
    #     stride = 2
    #     shortcut = ops.conv2d(shortcut, channels, 1, 1, stride, stride, name=name + "_shortcut", use_sn=use_sn)
    #     shortcut = tf.layers.batch_normalization(shortcut, training=True)

    # conv1 = ops.conv2d(input, channels, 3, 3, 1, 1, name=name + "_conv1", use_sn=use_sn)
    # conv1 = tf.layers.batch_normalization(conv1, training=True)
    # conv1 = tf.nn.leaky_relu(conv1)

    # conv2 = ops.conv2d(conv1, channels, 3, 3, stride, stride, name=name + "_conv2", use_sn=use_sn)
    # conv2 = tf.layers.batch_normalization(conv2, training=True)

    # conv2 += shortcut
    # output = tf.nn.leaky_relu(conv2)

    # return output

def simple_generator_block(x, out_channels, name):
    with tf.variable_scope(name):
        skip = x

        x = tf.contrib.layers.batch_norm(x, is_training=True, scope="bn1")
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, out_channels, (3, 3), strides=(2, 2), padding="same", name="conv1", data_format="channels_first")

        x = tf.contrib.layers.batch_norm(x, is_training=True, scope="bn2")
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_channels, (3, 3), padding="same", name="conv2", data_format="channels_first")

        skip = tf.layers.conv2d_transpose(skip, out_channels, (1, 1), strides=(2, 2), padding="same", data_format="channels_first") 

        return skip + x

def simple_discriminator_block(x, out_channels, name):
    with tf.variable_scope(name):
        skip = x

        x = tf.contrib.layers.batch_norm(x, is_training=True, scope="bn1")
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_channels, (3, 3), strides=(2, 2), padding="same", name="conv1", data_format="channels_first")

        x = tf.contrib.layers.batch_norm(x, is_training=True, scope="bn2")
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, out_channels, (3, 3), padding="same", name="conv2", data_format="channels_first")

        skip = tf.layers.conv2d(skip, out_channels, (1, 1), strides=(2, 2), padding="same", data_format="channels_first") 

        return skip + x

def class_conditional_generator_block(x, labels, out_channels, num_classes, is_training, name):
    with tf.variable_scope(name):
        bn0 = ops.ConditionalBatchNorm(num_classes, name='cbn_0')
        bn1 = ops.ConditionalBatchNorm(num_classes, name='cbn_1')
        x_0 = x
        x = tf.nn.relu(bn0(x, labels, is_training))
        x = resnet_architecture.get_conv(x, None, out_channels, "up", 'snconv1', True)
        # x = usample(x)
        # x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv1')
        x = tf.nn.relu(bn1(x, labels, is_training))
        x = resnet_architecture.get_conv(x, None, out_channels, "none", 'snconv2', True)
        # x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, name='snconv2')

        x_0 = resnet_architecture.get_conv(x_0, None, out_channels, "up", 'snconv3', True)
        # x_0 = usample(x_0)
        # x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, name='snconv3')

        return x_0 + x
