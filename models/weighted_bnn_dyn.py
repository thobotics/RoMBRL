# -*- coding: utf-8 -*-

"""
    bnn_dyn.py
    
    Created on  : February 03, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

#  Imports {{{ #

from collections import deque
import itertools
import logging
from time import time
import gc
import numpy as np
import tensorflow as tf

from lib.pysgmcmc.pysgmcmc.models.base_model import (
    BaseModel,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)

from lib.pysgmcmc.pysgmcmc.sampling import Sampler
from lib.pysgmcmc.pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

from lib.utils.data_batches import generate_weighted_batches
from lib.pysgmcmc.pysgmcmc.tensor_utils import uninitialized_params
from lib.me_trpo.utils import minimize_and_clip
from lib.utils.tf_distributions import exp_normalize

#  }}}  Imports #


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


#  Priors {{{ #

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


#  }}} Priors #

class BayesNeuralNetDynModel(object):
    def __init__(self, session, n_inputs, n_outputs, tf_scope="default",
                 sampling_method=Sampler.SGHMC,
                 get_net=get_default_net,
                 batch_generator=generate_weighted_batches,
                 batch_size=20, n_units=50, activation=tf.tanh,
                 n_nets=100, scale=1.0, a0=1.0, b0=0.1, a1=0.1, b1=0.1,
                 normalize_input=None, normalize_output=None,
                 seed=None, dtype=tf.float64, **sampler_kwargs):

        # Sanitize inputs
        assert isinstance(n_inputs, int)
        assert isinstance(n_outputs, int)
        assert isinstance(n_nets, int)
        assert isinstance(n_units, int)
        assert isinstance(batch_size, int)
        assert isinstance(dtype, tf.DType)

        assert n_inputs > 0
        assert n_outputs > 0
        assert n_nets > 0
        assert n_units > 0
        assert batch_size > 0

        assert callable(get_net)
        assert callable(batch_generator)

        # assert hasattr(stepsize_schedule, "update")
        # assert hasattr(stepsize_schedule, "__next__")

        if not Sampler.is_supported(sampling_method):
            raise ValueError(
                "'BayesianNeuralNetwork.__init__' received unsupported input "
                "for parameter 'sampling_method'. Input was: {input}.\n"
                "Supported sampling methods are enumerated in "
                "'Sampler' enum type.".format(input=sampling_method)
            )

        self.tf_scope = tf_scope

        self.sampling_method = sampling_method
        # self.stepsize_schedule = stepsize_schedule

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

        self.sampler_kwargs = sampler_kwargs

        self.samples = deque(maxlen=n_nets)

        self.seed = seed

        self.dtype = dtype

        self.session = session

        self.is_trained = False

        self.sampler = None

        self.log_weights = np.array([]).reshape((0, self.n_nets))

        self._initialize_variables(n_inputs, n_outputs, n_units)

    def get_variable(self, param_name):
        """ Public method to get a variable by name

        Parameters
        ----------
        param_name : Name of parameter.
        """

        return self.session.run([var for var in tf.global_variables(self.tf_scope)
                                 if var.op.name.split("/")[-1] == param_name][0])

    def _initialize_variables(self, n_inputs, n_outputs, n_units):
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
        self.Weight_Minibatch = tf.placeholder(shape=(None, self.n_nets),
                                          dtype=self.dtype,
                                          name="Weight_Minibatch")
        self.online_train = tf.placeholder_with_default(False, shape=[], name="online_train")
        self.n_datapoints = tf.placeholder(dtype=tf.int32, shape=[], name="n_datapoints")

        # setup params for covariances and neural network parameters

        # Diagonal covariance

        with tf.variable_scope(self.tf_scope):

            # self.log_Q = tf.Variable(
            #     np.log(np.random.gamma(self.a0, self.b0)) * tf.ones([1, n_outputs]), dtype=self.dtype,
            #     name="log_Q",
            # )  # default 2.0; 0.1

            # self.log_Q = tf.Variable(
            #     np.log([0.001]*2 + [0.1]*3 + [0.001]*2 + [0.1]*3 + [0.5]*2)[None], dtype=self.dtype,
            #     name="log_Q",
            # )  # default 2.0; 0.1

            # self.log_Q = tf.Variable(
            #     np.log(1.01) * tf.ones([1, n_outputs]), dtype=self.dtype,
            #     name="log_Q", #trainable=False,
            # )  # default 2.0; 0.1

            self.log_Q = tf.Variable(
                np.log(self.b0) * tf.ones([1, n_outputs]), dtype=self.dtype,
                name="log_Q",  # trainable=False,
            )  # default 2.0; 0.1

            self.log_lambda = tf.Variable(
                np.log(self.b1), dtype=self.dtype,
                name="log_lambda",  # trainable=False,
            )

            # self.log_lambda = tf.Variable(
            #     np.log(np.random.gamma(self.a1, self.b1)), dtype=self.dtype,
            #     name="log_lambda",  # trainable=False,
            # )

            # self.log_Q = tf.clip_by_value(self.log_Q, -20., 20.)
            # self.log_lambda = tf.clip_by_value(self.log_lambda, -20., 20.)

            # Normalize input
            if self.norm_input is not None:
                X_input = zero_mean_unit_var_normalization(self.X_Minibatch,
                                                        self.norm_input.mean, self.norm_input.std)[0]
            else:
                X_input = self.X_Minibatch

            net_output = self.get_net(inputs=X_input, n_outputs=n_outputs, n_units=n_units,
                                      activation=self.activation, seed=self.seed, dtype=self.dtype)

            # Unnormalize output
            if self.norm_output is not None:
                net_output = zero_mean_unit_var_unnormalization(net_output,
                                            self.norm_output.mean, self.norm_output.std)

            self.f_output = self.X_Minibatch[:, :n_outputs] + net_output

        self.network_params = tf.trainable_variables(self.tf_scope)

        # Initialize Tensorflow
        self.session.run(tf.variables_initializer(tf.global_variables(self.tf_scope)))

        """ Create optimizers """

        self.Nll, self.Mse = 0., 0.
        self._adam_opt, self._adam_op_opt = None, None

        # set up tensors for negative log likelihood and mean squared error

        self.Nll = self.negative_log_likelihood(
            X=self.X_Minibatch, Y=self.Y_Minibatch
        )

        self.Mse = self.mean_square_error(
            X=self.X_Minibatch, Y=self.Y_Minibatch
        )

        self.wKL = self.replay_posterior(
            X=self.X_Minibatch, Y=self.Y_Minibatch
        )

        with tf.variable_scope('adam_' + self.tf_scope):
            _prediction_opt = tf.train.AdamOptimizer(learning_rate=1e-3)

            # Normal Adam
            # prediction_opt_op.append(_prediction_opt.minimize(self.Nll[i]))

            # Clipped Adam
            self._adam_op_opt = minimize_and_clip(_prediction_opt,
                                                   self.Nll,
                                                   var_list=tf.get_collection(
                                                       tf.GraphKeys.TRAINABLE_VARIABLES,
                                                       scope=self.tf_scope),
                                                   collect_summary=True)

        # Initialize all variables
        _dynamics_adam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='adam_' + self.tf_scope)
        self.dynamics_adam_init = tf.variables_initializer(_dynamics_adam_vars)
        logging.debug('num_%s_adam_variables %d' % (self.tf_scope, len(_dynamics_adam_vars)))

        self.session.run(self.dynamics_adam_init)

        """ Predict dynamics """
        self.pred_output = []

        for i in range(self.n_nets):

            model_scope = "%s/model_%d" % ("predict_dynamics", i)

            with tf.variable_scope(model_scope):

                net_output = self.get_net(inputs=X_input, n_outputs=n_outputs, n_units=n_units,
                                          activation=self.activation, seed=self.seed, dtype=self.dtype)

                # Unnormalize output
                if self.norm_output is not None:
                    net_output = zero_mean_unit_var_unnormalization(net_output,
                                                                    self.norm_output.mean, self.norm_output.std)

                self.pred_output.append(self.X_Minibatch[:, :n_outputs] + net_output)

        self.pred_network_params = tf.global_variables("predict_dynamics")
        # self.assign_pred_params = [None for _ in self.pred_network_params]
        self.assign_pred_params = deque(maxlen=len(self.pred_network_params))

        self.session.run(tf.variables_initializer(self.pred_network_params))


    def _create_optimizer(self, X, y, step_size, mdecay=0.05, burn_in_steps=1000):
        """ Create loss using input datapoints `X`
            with corresponding labels `y`.

        Parameters
        ----------
        X : numpy.ndarray (N, D)
            Input training datapoints.

        y : numpy.ndarray (N,)
            Input training labels.

        """

        # self.X, self.y = X, y
        # n_datapoints, n_inputs = self.X.shape

        # # set up tensors for negative log likelihood and mean squared error
        # self.Nll = self.negative_log_likelihood(
        #     X=self.X_Minibatch, Y=self.Y_Minibatch,
        # )
        #
        # self.Mse = self.mean_square_error(
        #     X=self.X_Minibatch, Y=self.Y_Minibatch
        # )

        # Remove any leftover samples from previous "train" calls
        self.samples.clear()

        # Init sampler

        # Update network_params
        # del_idx = []
        #
        # for i in range(len(self.network_params)):
        #     if self.network_params[i].op.name.split("/")[-1] in ['log_Q', 'log_R', 'log_gamma', 'log_lambda']:
        #         del_idx.append(i)
        #
        # self.network_params = [i for j, i in enumerate(self.network_params) if j not in del_idx]

        if type(X) == tuple:
            generator = self.batch_generator(
                x=X[0], x_new=X[1], x_placeholder=self.X_Minibatch,
                y=y[0], y_new=y[1], y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                online_placeholder=self.online_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            n_datapoints = X[0].shape[0] + X[1].shape[0]
            n_inputs = X[0].shape[1]

        else:
            generator = self.batch_generator(
                x=X, x_placeholder=self.X_Minibatch,
                y=y, y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                online_placeholder=self.online_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            n_datapoints, n_inputs = X.shape

        self.sampler_kwargs.update({
            "tf_scope": self.tf_scope,
            "params": self.network_params,
            "cost_fun": lambda *_: self.Nll,
            "batch_generator": generator,
            "session": self.session,
            "seed": self.seed,
            "dtype": self.dtype,
            "stepsize_schedule": ConstantStepsizeSchedule(step_size),
        })

        # Update sampler
        if Sampler.is_burn_in_mcmc(self.sampling_method):
            # Not always used, only for `pysgmcmc.sampling.BurnInMCMCSampler`
            # subclasses.
            self.sampler_kwargs.update({
                "scale_grad": self.scale * n_datapoints,
                "mdecay": mdecay,
                "burn_in_steps": burn_in_steps,
            })

        # NOTE: Burn_in_steps might not be a necessary parameter anymore,
        # if we find that some samplers do not need it.
        # In this case, we might get rid of it and make users specify it
        # as part of `sampler_args` instead.

        # Update or create sampler

        if self.sampler is None:

            self.sampler = Sampler.get_sampler(
                self.sampling_method, **self.sampler_kwargs
            )

            # Extract all uninitialized parameters until this time
            uninit_params = uninitialized_params(
                session=self.session,
                params=tf.global_variables()
            )

            # Extract extra parameters
            self.reinit_params = self.sampler.vectorized_params + [self.sampler.epsilon] + uninit_params

            # Useful for quickly retrieving params later
            self.all_params = self.network_params + self.reinit_params

            # Initialize uninitialized parameters which is generated by sampler.
            init = tf.variables_initializer(uninit_params)

        else:

            # Reinitialize sampler parameters
            self.sampler.__init__(**self.sampler_kwargs)
            init = tf.variables_initializer(self.reinit_params)  # Force to reinitialize extra params

        self.session.run(init)

    def mean_square_error(self, X, Y):
        f_mean = self.f_output

        y_diff = Y - f_mean
        mse = tf.reduce_sum(tf.square(y_diff), axis=1)

        return tf.reduce_mean(mse)

    def negative_log_likelihood(self, X, Y):
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

        f_mean = self.f_output

        n_datapoints = tf.cast(self.n_datapoints, self.dtype)

        # Diagonal covariance
        log_Q = self.log_Q
        log_lambda = self.log_lambda

        # Inverse with diagonal matrix
        inv_Q = 1. / (tf.exp(log_Q) + 1e-16)
        inv_lambda = 1. / (tf.exp(log_lambda) + 1e-16)

        # Construct prior
        log_prior_w, log_prior_lambda = weight_prior(
            self.network_params, log_lambda, inv_lambda,
            self.a1, self.b1, self.dtype
        )

        # Define the log posterior distribution
        logging.info("Non-SSM Running")

        def kl_posterior(Y):
            y_diff = (Y[:self.batch_size] - f_mean[:self.batch_size],
                      Y[self.batch_size:] - f_mean[self.batch_size:])

            # Old data
            log_lik_data_old = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[0], inv_Q), y_diff[0]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))

            # weight = tf.exp(self.Weight_Minibatch)  # N x n_nets
            # norm_weight = weight / tf.reduce_sum(weight, axis=1, keep_dims=True)  # N x n_nets

            norm_weight = exp_normalize(self.Weight_Minibatch, axis=1)

            log_prev_data_old = self.Weight_Minibatch

            kl_old = log_prev_data_old - tf.reshape(log_lik_data_old, (-1, 1))  # N x n_nets
            kl_old = tf.reduce_sum(norm_weight*kl_old, axis=1)
            kl_old = tf.reduce_mean(kl_old)

            # New data
            log_lik_data_new = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[1], inv_Q), y_diff[1]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))
            log_lik_data_new = tf.reduce_mean(log_lik_data_new)

            return log_lik_data_new - kl_old

        def weighted_posterior(Y):
            y_diff = (Y[:self.batch_size] - f_mean[:self.batch_size],
                      Y[self.batch_size:] - f_mean[self.batch_size:])

            log_lik_data_old = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[0], inv_Q), y_diff[0]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))

            log_lik_data_new = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff[1], inv_Q), y_diff[1]), axis=1) +
                    tf.reduce_sum(log_Q, axis=1))

            log_lik_data = 1e-0 * tf.reduce_mean(log_lik_data_old) + \
                           tf.reduce_mean(log_lik_data_new)

            return log_lik_data

        def normal_posterior(Y):
            y_diff = Y - f_mean

            log_lik_data = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
                                                                                                           axis=1))
            return tf.reduce_mean(log_lik_data)

        log_lik_data = tf.cond(self.online_train,  # tf.shape(Y)[0] > self.batch_size,
                               lambda: kl_posterior(Y),
                               # lambda: weighted_posterior(Y),
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
        # log_posterior = tf.reduce_mean(log_lik_data) + log_prior_data / n_datapoints
        # log_posterior += log_prior_w / n_datapoints + log_prior_lambda / n_datapoints

        return -log_posterior

    def replay_posterior(self, X, Y):
        f_mean = self.f_output

        n_datapoints = tf.cast(self.n_datapoints, self.dtype)

        # Diagonal covariance
        log_Q = self.log_Q
        log_lambda = self.log_lambda

        # Inverse with diagonal matrix
        inv_Q = 1. / (tf.exp(log_Q) + 1e-16)
        inv_lambda = 1. / (tf.exp(log_lambda) + 1e-16)

        # Construct prior
        # log_prior_w, log_prior_lambda = weight_prior(
        #     self.network_params, log_lambda, inv_lambda,
        #     self.a1, self.b1, self.dtype
        # )

        # Compute replay posterior on old data
        y_diff = Y - f_mean

        log_lik_data = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
                                                                                                           axis=1))
        # log_prior_data = (1 - self.a0) * tf.reduce_sum(log_Q) - self.b0 * tf.reduce_sum(inv_Q)

        log_posterior = log_lik_data  #+ log_prior_data / n_datapoints
        # log_posterior += log_prior_w / n_datapoints + log_prior_lambda / n_datapoints

        return log_posterior

    def _compute_weights(self, X, y):

        if type(X) == tuple:
            x_train = np.vstack([X[0], X[1]])
            y_train = np.vstack([y[0], y[1]])

        else:
            x_train = X
            y_train = y

        self.log_weights = []

        for params in self.samples:

            feed_dict = dict(zip(self.network_params, params))
            feed_dict[self.X_Minibatch] = x_train
            feed_dict[self.Y_Minibatch] = y_train

            self.log_weights.append(self.session.run(self.wKL, feed_dict=feed_dict))

        self.log_weights = np.array(self.log_weights).T

        return

    def train_normal(self, X, y, X_val=None, y_val=None, step_size=1e-3, max_iters=8000):
        start_time = time()

        """ Create optimizer """

        # self.X, self.y = X, y

        # Reinitialize adam
        logging.info("Reinitialize dynamics Adam")
        self.session.run(self.dynamics_adam_init)

        logging.info("Starting Training")

        if type(X) == tuple:
            generator = self.batch_generator(
                x=X[0], x_new=X[1], x_placeholder=self.X_Minibatch,
                y=y[0], y_new=y[1], y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                online_placeholder=self.online_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = np.vstack([X[0], X[1]])
            y_batch = np.vstack([y[0], y[1]])

            n_train_datapoints = X[0].shape[0] + X[1].shape[0]

        else:
            generator = self.batch_generator(
                x=X, x_placeholder=self.X_Minibatch,
                y=y, y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                online_placeholder=self.online_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = X
            y_batch = y

            n_train_datapoints = X.shape[0]

        def log_full_training_error(iteration_index):

            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: x_batch,
                    self.Y_Minibatch: y_batch,
                    self.Weight_Minibatch: self.log_weights,
                    self.n_datapoints: n_train_datapoints
                }
            )
            total_nll_val, total_mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val,
                    self.Y_Minibatch: y_val,
                    self.Weight_Minibatch: self.log_weights,
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
            batch_dict = next(generator)

            _, training_loss = self.session.run([self._adam_op_opt, self.Nll], batch_dict)
            if j % 250 == 0:
                log_full_training_error(j)

        self.is_trained = True

    def train(self, X, y, X_val=None, y_val=None, step_size=1e-2, mdecay=0.05, burn_in_steps=1000, sample_steps=100, n_iters=50000, *args, **kwargs):
        """ Train our Bayesian Neural Network using input datapoints `X`
            with corresponding labels `y`.

        Parameters
        ----------
        X : numpy.ndarray (N*T+1, D)
            Input training datapoints.

        y : numpy.ndarray (N*T,)
            Input training labels.

        n_iters: int, optional
            Total number of iterations of the sampler to perform.
            Defaults to `50000`

        burn_in_steps: int, optional
            Number of burn-in steps to perform
            Defaults to `1000`.

        sample_steps: int, optional
            Number of sample steps to perform.
            Defaults to `100`.
        """

        assert isinstance(n_iters, int)
        assert isinstance(burn_in_steps, int)
        assert isinstance(sample_steps, int)

        assert n_iters > 0
        assert burn_in_steps >= 0
        assert sample_steps > 0

        start_time = time()

        self._create_optimizer(X, y, step_size=step_size, mdecay=mdecay, burn_in_steps=burn_in_steps)

        logging.info("Starting sampling")

        def log_full_training_error(iteration_index, is_sampling: bool):
            """ Compute the error of our last sampled network parameters
                on the full training dataset and use `logging.info` to
                log it. The boolean flag `sampling` is used to determine
                whether we are already collecting sampled networks, in which
                case additional info is logged using `logging.info`.

            Parameters
            ----------
            is_sampling : bool
                Boolean flag that specifies if we are already
                collecting samples or if we are still doing burn-in steps.
                If set to `True` we will also log the total number
                of samples collected thus far.

            """

            if type(X) == tuple:
                n_train_datapoints = X[0].shape[0] + X[1].shape[0]
                x_batch = np.vstack([X[0], X[1]])
                y_batch = np.vstack([y[0], y[1]])
            else:
                n_train_datapoints = X.shape[0]
                x_batch = X
                y_batch = y

            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: x_batch,
                    self.Y_Minibatch: y_batch,
                    self.Weight_Minibatch: self.log_weights,
                    self.n_datapoints: n_train_datapoints
                }
            )

            total_nll_val, total_mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val,
                    self.Y_Minibatch: y_val,
                    self.Weight_Minibatch: self.log_weights,
                    self.n_datapoints: X_val.shape[0]
                }
            )

            seconds_elapsed = time() - start_time
            if is_sampling:
                logging.info("Iter {:8d} : \n"
                             "\tNLL     = {} \n"
                             "\tNLL_val = {} \n"
                             "\tMSE     = {} \n"
                             "\tMSE_val = {} \n"
                             "Samples = {} \n"
                             "Time = {:5.2f}".format(
                    iteration_index, total_nll, total_nll_val,
                    total_mse, total_mse_val, len(self.samples), seconds_elapsed))
            else:
                logging.info("Iter {:8d} : \n"
                             "\tNLL     = {} \n"
                             "\tNLL_val = {} \n"
                             "\tMSE     = {} \n"
                             "\tMSE_val = {} \n"
                             "Time = {:5.2f}".format(
                    iteration_index, total_nll, total_nll_val,
                    total_mse, total_mse_val, seconds_elapsed))

        logging_intervals = {"burn-in": 512, "sampling": sample_steps}

        sample_chain = itertools.islice(self.sampler, n_iters)

        for iteration_index, (parameter_values, _) in enumerate(sample_chain):

            burning_in = iteration_index <= burn_in_steps

            # Q_new, lambda_new = self.get_variable("log_Q"), self.get_variable("log_lambda")
            # print(Q_new, "\n", "\n", lambda_new)

            if burning_in and iteration_index % logging_intervals["burn-in"] == 0:
                log_full_training_error(
                    iteration_index=iteration_index, is_sampling=False
                )

            if not burning_in and iteration_index % logging_intervals["sampling"] == 0:
                log_full_training_error(
                    iteration_index=iteration_index, is_sampling=True
                )

                # collect sample
                self.samples.append(parameter_values)

                if len(self.samples) == self.n_nets:
                    logging.info("Collect enough samples {}".format(len(self.samples)))
                    # collected enough sample networks, stop iterating
                    break

        self.is_trained = True

        # Compute log likelihood and predict params of all models
        self._compute_weights(X, y)
        self.feed_pred_params()

        # # Moment matching
        # self.samples_mean = []
        # self.samples_std = []
        # for i in range(len(self.samples[0])):
        #     params_mean = np.mean(np.array(self.samples)[:, i], axis=0)
        #     params_std = np.std(np.array(self.samples)[:, i], axis=0)
        #     self.samples_mean.append(params_mean)
        #     self.samples_std.append(params_std)

    def compute_network_output(self, params, input_data):
        """ Compute and return the output of the network when
            parameterized with `params` on `input_data`.

        Parameters
        ----------
        params : list of ndarray objects
            List of parameter values (ndarray)
            for each tensorflow.Variable parameter of our network.

        input_data : ndarray (N, D)
            Input points to compute the network output for.

        Returns
        -------
        network_output: ndarray (N,)
            Output of the network parameterized with `params`
            on the given `input_data` points.
        """

        feed_dict = dict(zip(self.network_params, params))
        feed_dict[self.X_Minibatch] = input_data
        return self.session.run(self.f_output, feed_dict=feed_dict)

    def compute_pred_network_output(self, model_idx, input_data):
        """ Compute and return the output of the network when
            parameterized with `params` on `input_data`.

        Parameters
        ----------
        params : list of ndarray objects
            List of parameter values (ndarray)
            for each tensorflow.Variable parameter of our network.

        input_data : ndarray (N, D)
            Input points to compute the network output for.

        Returns
        -------
        network_output: ndarray (N,)
            Output of the network parameterized with `params`
            on the given `input_data` points.
        """

        feed_dict = {}
        feed_dict[self.X_Minibatch] = input_data
        return self.session.run(self.pred_output[model_idx], feed_dict=feed_dict)

    def feed_pred_params(self):

        i = 0

        self.assign_pred_params.clear()

        for sample in self.samples:
            net_param = sample[2:]  # remove logQ and log_lambda

            for param_val in net_param:
                self.assign_pred_params.append(self.pred_network_params[i].assign(param_val))
                # self.assign_pred_params[i] = self.pred_network_params[i].assign(param_val)
                i += 1

        self.session.run(list(self.assign_pred_params))[0]
        # self.session.run(self.assign_pred_params)[0]

        # gc.collect()

    @BaseModel._check_shapes_predict
    def predict(self, X_test, normal=False, return_individual_predictions=True, model_idx=None, *args, **kwargs):
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

        if normal or len(self.samples) == 0:
            return self.session.run(self.f_output, feed_dict={self.X_Minibatch: X_test}), None
        elif return_individual_predictions:
            # Random return
            if model_idx is None:
                index = np.random.randint(self.n_nets)
            else:
                index = model_idx

            # sample = []
            # for i in range(len(self.samples[0])):
            #     mu = self.samples_mean[i]
            #     sigma = self.samples_std[i]
            #     eps = np.random.normal(0., 1., mu.shape)
            #     sample.append(sigma*eps + mu)
            # return self.compute_network_output(params=sample, input_data=X_test), None

            # return self.compute_network_output(params=self.samples[index], input_data=X_test), None
            return self.compute_pred_network_output(model_idx=index, input_data=X_test), None
        else:
            f_out = []

            for sample in self.samples:
                out = self.compute_network_output(params=sample, input_data=X_test)
                f_out.append(out)

            f_out = np.asarray(f_out)
            mean_prediction = np.mean(f_out, axis=0)

            # Total variance
            # v = np.mean(f_out ** 2 + theta_noise, axis=0) - m ** 2
            variance_prediction = np.mean((f_out - mean_prediction) ** 2, axis=0)

            return mean_prediction, variance_prediction

