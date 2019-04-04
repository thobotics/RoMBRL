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
import numpy as np
import tensorflow as tf

from lib.pysgmcmc.pysgmcmc.models.base_model import (
    BaseModel,
    zero_mean_unit_var_normalization,
    zero_mean_unit_var_unnormalization
)

from lib.pysgmcmc.pysgmcmc.sampling import Sampler
from lib.pysgmcmc.pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

from lib.utils.data_batches import generate_batches, generate_z_batches
from lib.pysgmcmc.pysgmcmc.tensor_utils import uninitialized_params
from lib.utils.tensor_adapter import repeat_v2
from lib.utils.tf_distributions import *

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
                 batch_generator=generate_batches,
                 batch_size=20, n_units=50, activation=tf.tanh,
                 n_nets=100, scale=1.0, a0=1.0, b0=0.1, a1=0.1, b1=0.1,
                 stochastic=True, n_samples=20, z_size=1,
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

        self.stochastic = stochastic
        self.n_samples = n_samples
        self.z_size = z_size

        self._initialize_variables(n_inputs, n_outputs, n_units)

    def get_variable(self, param_name):
        """ Public method to get a variable by name

        Parameters
        ----------
        param_name : Name of parameter.
        """

        return self.session.run([var for var in tf.global_variables(self.tf_scope)
                                 if var.op.name.split("/")[-1] == param_name][0])

    def logistic(self, x):
        logi = 1.0 / (1.0 + tf.exp(-x))
        return logi

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

        self.tf_n_samples = tf.placeholder_with_default(self.n_samples, None)

        self.X_rep = repeat_v2(self.X_Minibatch, self.tf_n_samples)
        self.Y_rep = repeat_v2(self.Y_Minibatch, self.tf_n_samples)

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
                name="log_Q",  #trainable=False,
            )  # default 2.0; 0.1

            self.log_lambda = tf.Variable(
                np.log(self.b1), dtype=self.dtype,
                name="log_lambda",  #trainable=False,
            )

            self.mean_z = tf.Variable(
                tf.zeros([1, self.z_size]), dtype=self.dtype,
                name="mean_z",  #trainable=False,
            )

            self.log_var_z = tf.Variable(
                np.log(0.5) * tf.ones([1, self.z_size]), dtype=self.dtype,
                name="log_var_z",  #trainable=False,
            )

            # self.log_lambda = tf.Variable(
            #     np.log(np.random.gamma(self.a1, self.b1)), dtype=self.dtype,
            #     name="log_lambda",  # trainable=False,
            # )

            self.log_Q = tf.clip_by_value(self.log_Q, -20., 20.)
            self.log_var_z = tf.clip_by_value(self.log_var_z, -20., 20.)
            self.log_lambda = tf.clip_by_value(self.log_lambda, -20., 20.)

            self.z_dist = DiagonalGaussian(tf.zeros([tf.shape(self.X_rep)[0], self.z_size]),
                                           tf.zeros([tf.shape(self.X_rep)[0], self.z_size]))  # Gaussian 0, 1
            # z_var = 1e-6 + self.logistic(self.log_var_z) * (n_inputs - 2e-6)
            self.z_input = self.z_dist.sample * tf.exp(self.log_var_z) + self.mean_z  # Reparameterization trick

            # Normalize input
            if self.norm_input is not None:
                X_input = zero_mean_unit_var_normalization(self.X_rep,
                                                        self.norm_input.mean, self.norm_input.std)[0]
            else:
                X_input = self.X_rep

            XZ_input = tf.concat([self.z_input, X_input], axis=1)
            net_output = self.get_net(inputs=XZ_input, n_outputs=n_outputs, n_units=n_units,
                                      activation=self.activation, seed=self.seed, dtype=self.dtype)

            # Unnormalize output
            if self.norm_output is not None:
                net_output = zero_mean_unit_var_unnormalization(net_output,
                                            self.norm_output.mean, self.norm_output.std)

            self.f_output = self.X_rep[:, :n_outputs] + net_output

        self.network_params = tf.trainable_variables(self.tf_scope)

        # Initialize Tensorflow
        self.session.run(tf.variables_initializer(tf.global_variables(self.tf_scope)))

    def _create_optimizer(self, X, y, step_size, mdecay=0.05):
        """ Create loss using input datapoints `X`
            with corresponding labels `y`.

        Parameters
        ----------
        X : numpy.ndarray (N, D)
            Input training datapoints.

        y : numpy.ndarray (N,)
            Input training labels.

        """
        self.X, self.y = X, y
        n_datapoints, n_inputs = self.X.shape

        # set up tensors for negative log likelihood and mean squared error
        # self.MM_Nll = self.mixture_log_likelihood(
        #     X=self.X_Minibatch, Z=self.Z_Minibatch, Y=self.Y_Minibatch, z_dist=z_dist
        # )

        self.Nll = self.negative_log_likelihood(
            X=self.X_rep, Y=self.Y_rep
        )

        self.Mse = self.mean_square_error(
            X=self.X_rep, Y=self.Y_rep
        )

        # Remove any leftover samples from previous "train" calls
        self.samples.clear()

        # Init sampler

        self.sampler_kwargs.update({
            "tf_scope": self.tf_scope,
            "params": self.network_params,
            "cost_fun": lambda *_: self.Nll,
            "batch_generator": self.batch_generator(
                x=self.X, x_placeholder=self.X_Minibatch,
                y=self.y, y_placeholder=self.Y_Minibatch,
                batch_size=self.batch_size,
                seed=self.seed
            ),
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
                "scale_grad": n_datapoints * self.scale,  #n_datapoints,
                "mdecay": mdecay,
                # "burn_in_steps": self.burn_in_steps,
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
        z_dist = self.z_dist
        Z = self.z_input

        f_mean = self.f_output

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

        y_diff = Y - f_mean
        log_py_x = -0.5 * (
                tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
                                                                                                       axis=1))
        log_prior_data = (1 - self.a0) * tf.reduce_sum(log_Q) - self.b0 * tf.reduce_sum(inv_Q)

        logps_Z = gaussian_diag_logps(self.mean_z, self.log_var_z, Z)
        logpz = tf.reduce_sum(logps_Z, axis=1)

        log_divergence = tf.reshape(log_py_x + logpz, [-1, self.n_samples])
        logF = logsumexp(log_divergence)

        log_posterior = tf.reduce_mean(logF - tf.log(tf.to_float(self.n_samples)))

        # log_posterior += log_prior_data
        log_posterior += log_prior_w + log_prior_lambda

        return -log_posterior

    def negative_log_likelihood_old(self, X, Y):
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
        y_diff = Y - f_mean

        log_lik_data = -0.5 * (
                    tf.reduce_sum(tf.multiply(tf.multiply(y_diff, inv_Q), y_diff), axis=1) + tf.reduce_sum(log_Q,
                                                                                                           axis=1))
        log_prior_data = (1 - self.a0) * tf.reduce_sum(log_Q) - self.b0 * tf.reduce_sum(inv_Q)

        # log_lik_data = -0.5 * (tf.reduce_sum(tf.square(y_diff), axis=1))
        # log_prior_data = 0.0

        log_posterior = tf.reduce_mean(log_lik_data) + log_prior_data
        log_posterior += log_prior_w + log_prior_lambda

        return -log_posterior

    def train_normal(self, X, y, X_val=None, y_val=None, step_size=1e-3, max_iters=8000):
        start_time = time()

        """ Create optimizer """

        self.X, self.y = X, y

        # set up tensors for negative log likelihood and mean squared error
        self.Nll = self.negative_log_likelihood(
            X=self.X_rep, Y=self.Y_rep
        )

        self.Mse = self.mean_square_error(
            X=self.X_rep, Y=self.Y_rep
        )

        from lib.me_trpo.utils import minimize_and_clip

        scope = self.tf_scope
        with tf.variable_scope('adam_' + scope):

            _prediction_opt = tf.train.AdamOptimizer(learning_rate=step_size)

            # Normal Adam
            # prediction_opt_op = _prediction_opt.minimize(self.Mse)

            # Clipped Adam
            prediction_opt_op = minimize_and_clip(_prediction_opt,
                                                self.Nll,
                                                var_list=tf.get_collection(
                                                         tf.GraphKeys.TRAINABLE_VARIABLES,
                                                         scope=self.tf_scope),
                                                collect_summary=True)

            # Initialize all variables
            _dynamics_adam_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='adam_' + scope)
            dynamics_adam_init = tf.variables_initializer(_dynamics_adam_vars)
            logging.debug('num_%s_adam_variables %d' % (scope, len(_dynamics_adam_vars)))

        self.session.run(dynamics_adam_init)

        logging.info("Starting Training")

        generator = self.batch_generator(
            x=self.X, x_placeholder=self.X_Minibatch,
            y=self.y, y_placeholder=self.Y_Minibatch,
            batch_size=self.batch_size,
            seed=self.seed
        )

        def log_full_training_error(iteration_index, is_sampling: bool):
            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: self.X,
                    self.Y_Minibatch: self.y
                }
            )
            total_nll_val, total_mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val,
                    self.Y_Minibatch: y_val
                }
            )
            seconds_elapsed = time() - start_time
            if is_sampling:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} MSE_val = {:.4e} "
                             "Time = {:5.2f}".format(iteration_index,
                                                     float(total_nll),
                                                     float(total_mse),
                                                     float(total_mse_val),
                                                     seconds_elapsed))
            else:
                logging.info("Iter {:8d} : NLL = {:.4e} MSE = {:.4e} MSE_val = {:.4e} "
                             "Samples = {} Time = {:5.2f}".format(
                    iteration_index, float(total_nll), float(total_mse), float(total_mse_val), len(self.samples),
                    seconds_elapsed))

        for j in range(0, max_iters + 1):
            batch_dict = next(generator)

            _, training_loss = self.session.run([prediction_opt_op, self.Nll], batch_dict)
            if j % 500 == 0:
                log_full_training_error(j, True)

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

        self._create_optimizer(X, y, step_size=step_size)

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
            total_nll, total_mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: self.X,
                    self.Y_Minibatch: self.y
                }
            )

            total_nll_val, total_mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val,
                    self.Y_Minibatch: y_val
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

        self.samples_mean = []
        for i in range(len(self.samples[0])):
            params = np.mean(np.array(self.samples)[:, i], axis=0)
            self.samples_mean.append(params)


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
        feed_dict[self.tf_n_samples] = 1
        return self.session.run(self.f_output, feed_dict=feed_dict)

    def forward_ssm(self, X_test):
        if not self.is_trained:
            raise ValueError(
                "Calling `bnn.predict()` on an untrained "
                "Bayesian Neural Network 'bnn' is not supported! "
                "Please call `bnn.train()` before calling `bnn.predict()`"
            )

        # Random chosen network
        index = np.random.randint(len(self.samples))
        out = self.compute_network_output(params=self.samples[index], input_data=X_test)

        log_Q = self.get_variable("log_Q").reshape(-1)

        Q = np.diag(np.exp(log_Q))

        # Generate noise
        process_noise = np.random.multivariate_normal(np.zeros(out.shape[1]), Q)
        out += process_noise

        return out, None

    @BaseModel._check_shapes_predict
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

        # return_individual_predictions = False

        if not self.is_trained:
            raise ValueError(
                "Calling `bnn.predict()` on an untrained "
                "Bayesian Neural Network 'bnn' is not supported! "
                "Please call `bnn.train()` before calling `bnn.predict()`"
            )

        if len(self.samples) == 0:
            return self.session.run(self.f_output, feed_dict={self.X_Minibatch: X_test, self.tf_n_samples: 1}), None
        elif return_individual_predictions:
            # Random return
            if model_idx is None:
                index = np.random.randint(self.n_nets)
            else:
                index = model_idx
            return self.compute_network_output(params=self.samples[index], input_data=X_test), None
            # return self.compute_network_output(params=self.samples_mean, input_data=X_test), None
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

