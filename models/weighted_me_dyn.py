# -*- coding: utf-8 -*-

"""
    me_dyn.py
    
    Created on  : February 26, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

from collections import deque
import itertools
import logging
from time import time
import numpy as np
import tensorflow as tf

from lib.pysgmcmc.pysgmcmc.models.base_model import (
    BaseModel,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)

from lib.utils.data_batches import generate_batches, generate_weighted_batches

from models.bnn_dyn import get_default_net, weight_prior
from lib.me_trpo.utils import minimize_and_clip

# TODO: Abstract this
class EnsembleNeuralNetDynModel(object):
    def __init__(self, session, n_inputs, n_outputs, tf_scope="default",
                 get_net=get_default_net,
                 batch_generator=generate_weighted_batches,
                 batch_size=20, n_units=50, activation=tf.tanh,
                 n_nets=5, scale=1.0, a0=1.0, b0=0.1, a1=0.1, b1=0.1,
                 normalize_input=None, normalize_output=None,
                 seed=None, dtype=tf.float64, **sampler_kwargs):

        self.tf_scope = tf_scope

        self.get_net = get_net
        self.batch_generator = batch_generator

        self.norm_input = normalize_input
        self.norm_output = normalize_output

        self.n_nets = n_nets
        self.activation = activation
        self.scale = scale
        self.a0 = a0
        self.b0 = b0
        self.a1 = a1
        self.b1 = b1
        self.batch_size = batch_size

        self.seed = seed

        self.dtype = dtype

        self.session = session

        self.is_trained = False

        self._adam_opt = []

        self._adam_op_opt = []

        self._initialize_variables(n_inputs, n_outputs, n_units, n_nets)

    def get_variable(self, param_name, model_idx):
        """ Public method to get a variable by name

        Parameters
        ----------
        param_name : Name of parameter.
        """
        model_scope = "%s/model_%d" % (self.tf_scope, model_idx)

        return self.session.run([var for var in tf.global_variables(model_scope)
                                 if var.op.name.split("/")[-1] == param_name][0])

    def _initialize_variables(self, n_inputs, n_outputs, n_units, n_nets):
        """ Initialize all trainable variables

        Parameters
        ----------
        n_inputs : Dimension of datapoints.
        """

        # set up placeholders for data minibatches
        self.X_Minibatch = tf.placeholder(shape=(None, n_inputs),
                                          dtype=self.dtype,
                                          name="X_Minibatch")
        self.Y_Minibatch = tf.placeholder(shape=(None, n_outputs),
                                          dtype=self.dtype,
                                          name="Y_Minibatch")
        self.online_train = tf.placeholder_with_default(False, shape=[], name="online_train")
        self.n_datapoints = tf.placeholder(dtype=tf.int32, shape=[], name="n_datapoints")

        # setup params for covariances and neural network parameters

        # Diagonal covariance
        self.log_Q = []
        self.log_lambda = []
        self.f_output = []
        self.network_params = []

        # Normalize input
        if self.norm_input is not None:
            X_input = zero_mean_unit_var_normalization(self.X_Minibatch,
                                                       self.norm_input.mean, self.norm_input.std)[0]
        else:
            X_input = self.X_Minibatch

        for i in range(n_nets):

            model_scope = "%s/model_%d" % (self.tf_scope, i)

            with tf.variable_scope(model_scope):

                # self.log_Q.append(tf.Variable(
                #     np.log(np.random.gamma(self.a0, self.b0)) * tf.ones([1, n_outputs]), dtype=self.dtype,
                #     name="log_Q",
                # ))  # default 2.0; 0.1

                self.log_Q.append(tf.Variable(
                    np.log(self.b0) * tf.ones([1, n_outputs]), dtype=self.dtype,
                    name="log_Q", #trainable=False,
                ))  # default 2.0; 0.1

                self.log_lambda.append(tf.Variable(
                    np.log(np.random.gamma(self.a1, self.b1)), dtype=self.dtype,
                    name="log_lambda", #trainable=False,
                ))

                net_output = self.get_net(inputs=X_input, n_outputs=n_outputs,  n_units=n_units,
                                          activation=self.activation, seed=self.seed, dtype=self.dtype)

                # Unnormalize output
                if self.norm_output is not None:
                    net_output = zero_mean_unit_var_unnormalization(net_output,
                                                self.norm_output.mean, self.norm_output.std)

                self.f_output.append(self.X_Minibatch[:, :n_outputs] + net_output)

                self.network_params.append(tf.trainable_variables(model_scope))

        # Initialize Tensorflow
        self.session.run(tf.variables_initializer(tf.global_variables(self.tf_scope)))

        """ Create optimizers """

        self.Nll, self.Mse = [], []

        # set up tensors for negative log likelihood and mean squared error
        for i in range(self.n_nets):
            self.Nll.append(self.negative_log_likelihood(
                X=self.X_Minibatch, Y=self.Y_Minibatch, model_idx=i
            ))

            self.Mse.append(self.mean_square_error(
                X=self.X_Minibatch, Y=self.Y_Minibatch, model_idx=i
            ))

            with tf.variable_scope('adam_' + self.tf_scope):
                _prediction_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

                # Normal Adam
                # prediction_opt_op.append(_prediction_opt.minimize(self.Nll[i]))

                # Clipped Adam
                self._adam_op_opt.append(minimize_and_clip(_prediction_opt,
                                                           self.Nll[i],
                                                           var_list=tf.get_collection(
                                                               tf.GraphKeys.TRAINABLE_VARIABLES,
                                                               scope=self.tf_scope),
                                                           collect_summary=True))

        # Initialize all variables
        _dynamics_adam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='adam_' + self.tf_scope)
        self.dynamics_adam_init = tf.variables_initializer(_dynamics_adam_vars)
        logging.debug('num_%s_adam_variables %d' % (self.tf_scope, len(_dynamics_adam_vars)))

        self.session.run(self.dynamics_adam_init)

    def mean_square_error(self, X, Y, model_idx=0):
        f_mean = self.f_output[model_idx]

        y_diff = Y - f_mean
        mse = tf.reduce_sum(tf.square(y_diff), axis=1)

        return tf.reduce_mean(mse)

    def negative_log_likelihood(self, X, Y, model_idx=0):
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

        f_mean = self.f_output[model_idx]

        n_datapoints = tf.cast(self.n_datapoints, self.dtype)  # tf.cast(tf.shape(Y)[0], self.dtype)*2

        # Diagonal covariance
        log_Q = self.log_Q[model_idx]
        log_lambda = self.log_lambda[model_idx]

        # Inverse with diagonal matrix
        inv_Q = 1. / (tf.exp(log_Q) + 1e-16)
        inv_lambda = 1. / (tf.exp(log_lambda) + 1e-16)

        # Construct prior
        log_prior_w, log_prior_lambda = weight_prior(
            self.network_params[model_idx], log_lambda, inv_lambda,
            self.a1, self.b1, self.dtype
        )

        # Define the log posterior distribution
        def weighted_posterior(Y):
            y_diff = (Y[:self.batch_size] - f_mean[:self.batch_size],
                      Y[self.batch_size:] - f_mean[self.batch_size:])

            log_lik_data_old = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[0], inv_Q), y_diff[0]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))

            log_lik_data_new = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[1], inv_Q), y_diff[1]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))

            log_lik_data = 1e-1 * tf.reduce_mean(log_lik_data_old) + \
                           tf.reduce_mean(log_lik_data_new)

            return log_lik_data

        def normal_posterior(Y):
            y_diff = Y - f_mean

            log_lik_data = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
                                                                                                           axis=1))
            return tf.reduce_mean(log_lik_data)

        log_lik_data = tf.cond(self.online_train,  # tf.shape(Y)[0] > self.batch_size,
                               lambda: weighted_posterior(Y),
                               lambda: normal_posterior(Y))

        log_prior_data = (1 - self.a0) * tf.reduce_sum(log_Q) - self.b0 * tf.reduce_sum(inv_Q)

        log_posterior = log_lik_data + log_prior_data / n_datapoints
        log_posterior += log_prior_w / n_datapoints + log_prior_lambda / n_datapoints

        # y_diff = Y - f_mean
        #
        # log_lik_data = -0.5 * (
        #             tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
        #                                                                                                    axis=1))
        # log_prior_data = (1 - self.a0) * tf.reduce_sum(log_Q) - self.b0 * tf.reduce_sum(inv_Q)
        #
        # # log_lik_data = -0.5 * (tf.reduce_sum(tf.square(y_diff), axis=1))
        # # log_prior_data = 0.0
        #
        # log_posterior = tf.reduce_mean(log_lik_data) + log_prior_data / n_datapoints
        # log_posterior += log_prior_w / n_datapoints + log_prior_lambda / n_datapoints

        return -log_posterior

    def train_normal(self, X, y, X_val=None, y_val=None, step_size=1e-3, max_iters=8000):
        start_time = time()

        """ Create optimizer """

        self.X, self.y = X, y

        if type(self.X) == tuple:
            generator = self.batch_generator(
                x=self.X[0], x_new=self.X[1], x_placeholder=self.X_Minibatch,
                y=self.y[0], y_new=self.y[1], y_placeholder=self.Y_Minibatch,
                online_placeholder=self.online_train,
                n_points_placeholder=self.n_datapoints,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = np.vstack([self.X[0], self.X[1]])
            y_batch = np.vstack([self.y[0], self.y[1]])

            n_train_datapoints = self.X[0].shape[0] + self.X[1].shape[0]

        else:
            generator = self.batch_generator(
                x=self.X, x_placeholder=self.X_Minibatch,
                y=self.y, y_placeholder=self.Y_Minibatch,
                online_placeholder=self.online_train,
                n_points_placeholder=self.n_datapoints,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = self.X
            y_batch = self.y

            n_train_datapoints = self.X.shape[0]

        # Reinitialize adam
        logging.info("Reinitialize dynamics Adam")
        self.session.run(self.dynamics_adam_init)

        logging.info("Start Training")

        def log_full_training_error(iteration_index):
            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: x_batch,
                    self.Y_Minibatch: y_batch,
                    self.n_datapoints: n_train_datapoints
                }
            )
            total_nll_val, total_mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val,
                    self.Y_Minibatch: y_val,
                    self.n_datapoints: X_val.shape[0]
                }
            )
            seconds_elapsed = time() - start_time

            logging.info("Iter {:8d} : \n"
                         "\tNLL     = {} \n"
                         "\tNLL_val = {} \n"
                         "\tMSE     = {} \n"
                         "\tMSE_val = {} \n"
                         "Time = {:5.2f}".format(
                iteration_index, total_nll, total_nll_val, total_mse, total_mse_val, seconds_elapsed))

        for j in range(0, max_iters + 1):

            for i in range(self.n_nets):
                batch_dict = next(generator)
                _, training_loss = self.session.run([self._adam_op_opt[i], self.Nll[i]], batch_dict)

            if j % 250 == 0:
                log_full_training_error(j)

        self.is_trained = True

    def predict(self, X_test, return_individual_predictions=True, model_idx=None, *args, **kwargs):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test datapoints.

        return_individual_predictions: bool
            If set to `True` than the individual predictions of
            all samples are returned.

        Returns
        ----------
        mean: np.array(N,)
            predictive mean

        variance: np.array(N,)
            predictive variance

        """

        if not self.is_trained:
            raise ValueError(
                "Calling `bnn.predict()` on an untrained "
                "Bayesian Neural Network 'bnn' is not supported! "
                "Please call `bnn.train()` before calling `bnn.predict()`"
            )

        if self.n_nets == 1:
            return self.session.run(self.f_output[0], feed_dict={self.X_Minibatch: X_test}), None
        elif return_individual_predictions:
            # Random return
            if model_idx is None:
                index = np.random.randint(self.n_nets)
            else:
                index = model_idx
            return self.session.run(self.f_output[index], feed_dict={self.X_Minibatch: X_test}), None
