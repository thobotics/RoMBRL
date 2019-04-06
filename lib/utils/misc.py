# -*- coding: utf-8 -*-

"""
    miscellaneous.py
    
    Created on  : April 06, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import logging
import tensorflow as tf


def get_logger(logger_name, folderpath, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    to_generate = [('info.log',logging.INFO), ('debug.log',logging.DEBUG)]
    for logname, handler_level in to_generate:
        # Create a file handler
        handler = logging.FileHandler(folderpath+'/' + logname)
        handler.setLevel(handler_level)

        # Create a logging format
        if logname == 'debug.log':
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def minimize_and_clip(optimizer, objective, var_list, clip_val=None, collect_summary=False):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            if clip_val is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
            if collect_summary:
                with tf.name_scope('%s/%s%s/gradients' % (objective.name[:-2], var.name[:-2], var.name[-1])):
                    variable_summaries(gradients[i][0])
                    tf.summary.scalar('norm', tf.norm(gradients[i]))
    return optimizer.apply_gradients(gradients)

