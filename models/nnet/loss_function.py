# -*- coding: utf-8 -*-

"""
    loss_function.py
    
    Created on  : April 05, 2019
        Author  : anonymous
        Name    : Anonymous
"""

import numpy as np
import tensorflow as tf
from lib.utils.tf_distributions import exp_normalize


def weight_prior(parameters, log_lambda, inv_lambda, a=1.0, b=0.01, dtype=tf.float64):
    """ Short description

    Parameters
    ----------
     : Description for argument
    """

    log_prior_w = tf.convert_to_tensor(0., name="param_reg", dtype=dtype)
    n_params = tf.convert_to_tensor(0., name="n_params", dtype=dtype)

    for parameter in parameters:
        if parameter.name not in ['log_Q', 'log_R', 'log_gamma', 'log_lambda']:
            log_prior_w += -0.5 * tf.reduce_sum(inv_lambda * tf.square(parameter) + log_lambda)
            n_params += tf.cast(
                tf.reduce_prod(tf.to_float(parameter.shape)), dtype=dtype
            )
    log_prior_w = log_prior_w / n_params
    log_prior_lambda = (1 - a) * log_lambda - b * inv_lambda

    return log_prior_w, log_prior_lambda


def mean_square_error(y_logit, y_true):
    y_diff = y_true - y_logit
    mse = tf.reduce_sum(tf.square(y_diff), axis=1)

    return tf.reduce_mean(mse)


def gaussian_log_likelihood(y_logit, y_true, log_var):

    # Inverse with diagonal matrix
    inv_var = 1. / (tf.exp(log_var) + 1e-16)

    y_diff = y_true - y_logit
    log_lik_data = -0.5 * (
            tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_var), y_diff), axis=1) +
            tf.reduce_sum(log_var, axis=1))

    return log_lik_data


class TrainingLoss(object):

    def __init__(self, y_logit, y_true, log_Q, log_lambda, a0, b0, a1, b1, network_params,
                 n_datapoints_placeholder, weight_placeholder, continual_train, continual_method="kl",
                 batch_size=100, dtype=tf.float32):

        assert continual_method in ["kl", "weight"]

        self.y_logit = y_logit
        self.y_true = y_true

        self.log_Q = log_Q
        self.log_lambda = log_lambda

        # Inverse with diagonal matrix
        self.inv_Q = 1. / (tf.exp(log_Q) + 1e-16)
        self.inv_lambda = 1. / (tf.exp(log_lambda) + 1e-16)

        self.a0 = a0
        self.b0 = b0
        self.a1 = a1
        self.b1 = b1
        self.network_params = network_params

        self.tf_n_datapoints = n_datapoints_placeholder
        self.tf_log_weight = weight_placeholder
        self.tf_continual_train = continual_train

        self.continual_method = continual_method
        self.batch_size = batch_size
        self.dtype = dtype

    def _kl_likelihood(self, y_logit, y_true):
        y_diff_old = y_true[:self.batch_size] - y_logit[:self.batch_size]
        y_diff_new = y_true[self.batch_size:] - y_logit[self.batch_size:]

        # Old data
        log_lik_data_old = -0.5 * (
                tf.reduce_sum(tf.multiply(tf.multiply(y_diff_old, self.inv_Q), y_diff_old), axis=1) +
                tf.reduce_sum(self.log_Q, axis=1))

        norm_weight = exp_normalize(self.tf_log_weight, axis=1)
        log_prev_data_old = self.tf_log_weight

        kl_old = log_prev_data_old - tf.reshape(log_lik_data_old, (-1, 1))  # N x n_nets
        kl_old = tf.reduce_sum(norm_weight * kl_old, axis=1)
        kl_old = tf.reduce_mean(kl_old)

        # New data
        log_lik_data_new = -0.5 * (
                tf.reduce_sum(tf.multiply(tf.multiply(y_diff_new, self.inv_Q), y_diff_new), axis=1) +
                tf.reduce_sum(self.log_Q, axis=1))
        log_lik_data_new = tf.reduce_mean(log_lik_data_new)

        return log_lik_data_new - kl_old

    def _weighted_likelihood(self, y_logit, y_true):
        y_diff_old = y_true[:self.batch_size] - y_logit[:self.batch_size]
        y_diff_new = y_true[self.batch_size:] - y_logit[self.batch_size:]

        log_lik_data_old = -0.5 * (
                tf.reduce_sum(tf.multiply(tf.multiply(y_diff_old, self.inv_Q), y_diff_old), axis=1) +
                tf.reduce_sum(self.log_Q, axis=1))

        log_lik_data_new = -0.5 * (
                tf.reduce_sum(tf.multiply(tf.multiply(y_diff_new, self.inv_Q), y_diff_new), axis=1) +
                tf.reduce_sum(self.log_Q, axis=1))

        log_lik_data = 1e-0 * tf.reduce_mean(log_lik_data_old) + tf.reduce_mean(log_lik_data_new)

        return log_lik_data

    def mse(self):
        return mean_square_error(self.y_logit, self.y_true)

    def log_likehood(self):
        return gaussian_log_likelihood(self.y_logit, self.y_true, self.log_Q)

    def negative_log_posterior(self):
        """ Compute the negative log likelihood of the
            current network parameters with respect to inputs `X` with
            labels `Y`.

        Parameters
        ----------
        X : tensorflow.Placeholder
            Placeholder for input datapoints.

        Y : tensorflow.Placeholder
            Placeholder for input labels.

        Returns
        -------
        neg_log_like: tensorflow.Tensor
            Negative log likelihood of the current network parameters with
            respect to inputs `X` with labels `Y`.


        mse: tensorflow.Tensor
            Mean squared error of the current network parameters
            with respect to inputs `X` with labels `Y`.

        """

        y_logit = self.y_logit
        y_true = self.y_true
        n_datapoints = tf.cast(self.tf_n_datapoints, self.dtype)

        # Construct prior
        log_prior_w, log_prior_lambda = weight_prior(
            self.network_params, self.log_lambda, self.inv_lambda,
            self.a1, self.b1, self.dtype
        )

        # Define the log posterior distribution

        if self.continual_method == "kl":
            weight_likelihood = self._kl_likelihood(y_logit, y_true)
        else:
            weight_likelihood = self._weighted_likelihood(y_logit, y_true)

        normal_likelihood = tf.reduce_mean(gaussian_log_likelihood(y_logit, y_true, self.log_Q))

        log_lik_data = tf.cond(self.tf_continual_train,
                               lambda *_: weight_likelihood,
                               lambda *_: normal_likelihood)

        log_prior_data = (1 - self.a0) * tf.reduce_sum(self.log_Q) - self.b0 * tf.reduce_sum(self.inv_Q)

        log_posterior = log_lik_data + log_prior_data / n_datapoints
        log_posterior += log_prior_w / n_datapoints + log_prior_lambda / n_datapoints

        return -log_posterior
