import resnet_architecture
import consts

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
