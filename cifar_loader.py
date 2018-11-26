import tensorflow as tf

def parse_images(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=3)
  image_flipped = tf.image.random_flip_left_right(image_decoded)
  image_normalized = 2.0 * tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
  image_normalized.set_shape([32, 32, 3])
  return image_normalized

def text_to_index(text_label):
    return tf.string_to_number(text_label, out_type=tf.int32)

def text_to_one_hot(text_label):
    int_label = tf.string_to_number(text_label, out_type=tf.int32)
    return tf.one_hot(int_label, 11)


def load_images_and_labels(batch_size, data_dir):
    image_files_dataset = tf.data.Dataset.list_files(data_dir + "train/*", shuffle=False)
    image_files_dataset = image_files_dataset.concatenate(tf.data.Dataset.list_files(data_dir + "test/*", shuffle=False))
    image_dataset = image_files_dataset.map(parse_images, num_parallel_calls=32)

    label_lines_dataset = tf.data.TextLineDataset([data_dir + "Train_cntk_text.txt", data_dir + "Test_cntk_text.txt"])
    label_dataset = label_lines_dataset.map(text_to_one_hot)
    index_dataset = label_lines_dataset.map(text_to_index)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset, index_dataset))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # dataset = dataset.cache()
    dataset = dataset.repeat()
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000))
    dataset = dataset.prefetch(batch_size)
    return dataset.make_one_shot_iterator()
