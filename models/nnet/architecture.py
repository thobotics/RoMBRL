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
def get_default_net(inputs, n_outputs=1, n_units=None, activation=None, seed=None, dtype=tf.float64):

    if not n_units:
        n_units = [50, 50]

    if not activation:
        activation = ["tf.tanh", "tf.tanh"]

    fc_layer = inputs
    for i in range(len(n_units)):
        fc_layer = tf.layers.dense(
            fc_layer, units=n_units[i], activation=eval(activation[i]),
            kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
            bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
            name="fc_layer_%d" % (i + 1)
        )

    output = tf.layers.dense(
        fc_layer, units=n_outputs, activation=None,  # linear activation
        kernel_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        bias_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
        name="output"
    )

    return output

#  }}} Default Network Architecture #
