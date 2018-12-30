import tensorflow as tf

def text_to_index(text_label):
    return tf.string_to_number(text_label, out_type=tf.int32)

def text_to_one_hot(text_label, categories):
    int_label = tf.string_to_number(text_label, out_type=tf.int32)
    return tf.one_hot(int_label, categories + 1)
