# -*- coding: utf-8 -*-

"""
    architecture.py
    
    Created on  : April 05, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import numpy as np
import tensorflow as tf

#  Default Network Architecture {{{ #

def get_default_net(inputs, n_outputs=1, n_units=50, activation=tf.tanh, seed=None, dtype=tf.float64):
    from tensorflow.contrib.layers import variance_scaling_initializer as HeNormal

    fc_layer_1 = tf.layers.dense(
        inputs, units=n_units, activation=activation,
        kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        # kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        # bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_1"
    )

    fc_layer_2 = tf.layers.dense(
        fc_layer_1, units=n_units, activation=activation,
        kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        # kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        # bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="fc_layer_2"
    )

    # fc_layer_3 = tf.layers.dense(
    #     fc_layer_2, units=n_units, activation=tf.tanh,  # tf.nn.relu
    #     kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
    #     bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
    #     # kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
    #     # bias_initializer=tf.zeros_initializer(dtype=dtype),
    #     name="fc_layer_3"
    # )

    output = tf.layers.dense(
        fc_layer_2, units=n_outputs, activation=None,  # linear activation
        kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        # kernel_initializer=HeNormal(factor=1.0, dtype=dtype, seed=seed),
        # bias_initializer=tf.zeros_initializer(dtype=dtype),
        name="output"
    )

    return output

#  }}} Default Network Architecture #
