import tensorflow as tf

from gan import GAN
import resnet_blocks
import ops
import loader

tf.compat.v1.disable_eager_execution()

image_feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


def parse_images(tfrecord):
    proto = tf.io.parse_single_example(tfrecord, image_feature_description)

    image_decoded = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image_normalized = 2.0 * \
        tf.image.convert_image_dtype(image_decoded, tf.float32) - 1.0
    image_flipped = tf.image.random_flip_left_right(image_normalized)
    image_nchw = tf.transpose(image_flipped, [2, 0, 1])

    raw_label = proto['label'] - 1
    one_hot_label = tf.one_hot(raw_label, 101)

    return image_nchw, one_hot_label, raw_label


def load_imagenet(batch_size):
    files = tf.data.Dataset.list_files("/mnt/Bulk Storage/imagenet/tfrecords/64x64/*")
    dataset = files.interleave(lambda f: tf.data.TFRecordDataset(f), cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(40000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf.compat.v1.data.make_one_shot_iterator(dataset)

G_DIM = 64
def resnet_generator(z, labels):
    with tf.compat.v1.variable_scope('generator'):
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

D_DIM = 64
def resnet_discriminator(x, labels, reuse=False, use_sn=True):
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
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

class IMAGENET_GAN(GAN):
    def __init__(self, training_steps, batch_size, label_based_discriminator, output_real_images=False):
        super().__init__(64, 64, "imagenet", training_steps, batch_size, 1000, output_real_images=True)

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
        yg = tf.reshape(tf.tile(tf.one_hot(1000, 1001), [batch_size]), [batch_size, 1001])

        return x, yx, yg

g = IMAGENET_GAN(500000, 100, True)
g.run()
