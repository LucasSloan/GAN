# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import ops

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.keras.initializers.glorot_normal, name='sn_conv1x1'):
  with tf.compat.v1.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.compat.v1.get_variable(
        'w', [k_h, k_w, input_.get_shape()[1], output_dim],
        initializer=init)
    w_bar = ops.spectral_norm(w)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME', data_format="NCHW")
    return conv

def sn_non_local_block_sim(x, update_collection, name, init=tf.keras.initializers.glorot_normal):
  with tf.compat.v1.variable_scope(name):
    _, num_channels, h, w = x.get_shape().as_list()
    location_num = h * w
    downsampled_num = location_num // 4

    # theta path
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta_t = tf.reshape(
        theta, [-1, num_channels // 8, location_num])

    # phi path
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.compat.v1.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2, data_format='channels_first')
    phi_t = tf.reshape(
        phi, [-1, num_channels // 8, downsampled_num])


    attn_t = tf.matmul(phi_t, theta_t, transpose_a=True)
    attn_t = tf.nn.softmax(attn_t)
    print(tf.reduce_sum(attn_t, axis=-1))

    # g path
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.compat.v1.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2, data_format='channels_first')
    g_t = tf.reshape(
      g, [-1, num_channels // 2, downsampled_num])

    attn_g_t = tf.matmul(g_t, attn_t)
    attn_g = tf.reshape(attn_g_t, [-1, num_channels // 2, h, w])
    sigma = tf.compat.v1.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')

    out = x + sigma * attn_g
    return out