# -*- coding: utf-8 -*-

"""
    tensor_adapter.py
    
    Created on  : February 03, 2019
        Author  : anonymous
        Name    : Anonymous
"""
import numpy as np
import tensorflow as tf


def smooth3d_to_bnn(tensor):
    """ Reshape smooth_x to fit bnn input

    Parameters
    ----------
     tensor : 3d tensor with shape (dx, N, T)
    """

    dx, Ns, T = tensor.shape
    result = tensor.reshape((dx, Ns * T)).T

    return result.reshape((Ns, T, dx))


def repeat_v2(x, k):
    """ repeat k times along first dimension """

    def change(x, k):
        shape = x.get_shape().as_list()[1:]
        x_1 = tf.expand_dims(x, 1)
        tile_shape = tf.concat([tf.ones(1, dtype='int32'), [k], tf.ones([tf.rank(x) - 1], dtype='int32')], axis=0)
        x_rep = tf.tile(x_1, tile_shape)
        new_shape = np.insert(shape, 0, -1)
        x_out = tf.reshape(x_rep, new_shape)
        return x_out

    return tf.cond(tf.equal(k, 1),
                   lambda: x,
                   lambda: change(x, k))

