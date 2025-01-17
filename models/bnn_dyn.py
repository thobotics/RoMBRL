# -*- coding: utf-8 -*-

"""
    bnn_dyn.py
    
    Created on  : February 03, 2019
        Author  : anonymous
        Name    : Anonymous
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

from lib.utils.data_batches import generate_weighted_batches
from lib.pysgmcmc.pysgmcmc.tensor_utils import uninitialized_params
from lib.utils.misc import minimize_and_clip
from models.nnet.architecture import get_default_net
from models.nnet.loss_function import TrainingLoss

#  }}}  Imports #


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
        assert isinstance(batch_size, int)
        assert isinstance(dtype, tf.DType)

        assert n_inputs > 0
        assert n_outputs > 0
        assert n_nets > 0
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

        self.log_weights = np.zeros((self.batch_size, self.n_nets))  # np.array([]).reshape((0, self.n_nets))

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
        self.continual_train = tf.placeholder_with_default(False, shape=[], name="continual_train")
        self.n_datapoints = tf.placeholder(dtype=tf.int32, shape=[], name="n_datapoints")

        # setup params for covariances and neural network parameters

        # Diagonal covariance

        with tf.variable_scope(self.tf_scope):

            self.log_Q = tf.Variable(
                np.log(self.b0) * tf.ones([1, n_outputs]), dtype=self.dtype,
                name="log_Q",
            )

            self.log_lambda = tf.Variable(
                np.log(self.b1), dtype=self.dtype,
                name="log_lambda",
            )

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

        self.training_loss = TrainingLoss(y_logit=self.f_output, y_true=self.Y_Minibatch,
                                          log_Q=self.log_Q, log_lambda=self.log_lambda,
                                          a0=self.a0, b0=self.b0, a1=self.a1, b1=self.b1,
                                          network_params=self.network_params,
                                          n_datapoints_placeholder=self.n_datapoints,
                                          weight_placeholder=self.Weight_Minibatch,
                                          continual_train=self.continual_train, continual_method="kl",
                                          batch_size=self.batch_size, dtype=self.dtype)

        self.Nll = self.training_loss.negative_log_posterior()

        self.Mse = self.training_loss.mse()

        self.log_likelihood = self.training_loss.log_likehood()

        """ Adam normal optimizer """

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

                pred_output = self.X_Minibatch[:, :n_outputs] + net_output
                # pred_output += tf.random_normal(shape=(tf.shape(pred_output))) * tf.exp(self.log_Q)
                self.pred_output.append(pred_output)

        self.pred_network_params = tf.global_variables("predict_dynamics")
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

        # Remove any leftover samples from previous "train" calls
        self.samples.clear()

        # Init sampler

        if type(X) == tuple:
            generator = self.batch_generator(
                x=X[0], x_new=X[1], x_placeholder=self.X_Minibatch,
                y=y[0], y_new=y[1], y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                continual_placeholder=self.continual_train,
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
                continual_placeholder=self.continual_train,
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

    def _compute_weights(self, X, y):

        if type(X) == tuple:
            x_train = np.vstack([X[0], X[1]])
            y_train = np.vstack([y[0], y[1]])
        else:
            x_train = X
            y_train = y

        i = 0

        self.log_weights = np.zeros((x_train.shape[0], self.n_nets))

        for params in self.samples:
            feed_dict = dict(zip(self.network_params, params))
            feed_dict[self.X_Minibatch] = x_train
            feed_dict[self.Y_Minibatch] = y_train

            self.log_weights[:, i] = self.session.run(self.log_likelihood, feed_dict=feed_dict)
            i += 1

    def log_full_training_error(self, x_batch, y_batch, X_val, y_val,
                                start_time, iteration_index, is_sampling: bool):

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

        log_batch = 2000
        n_tr = max(1, len(x_batch) // log_batch)
        n_va = max(1, len(X_val) // log_batch)

        total_nll, total_mse = 0., 0.
        total_nll_val, total_mse_val = 0., 0.

        for i in range(n_tr):
            nll, mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: x_batch[log_batch * i:log_batch * (i + 1)],
                    self.Y_Minibatch: y_batch[log_batch * i:log_batch * (i + 1)],
                    self.Weight_Minibatch: np.zeros((self.batch_size, self.n_nets)),
                    self.n_datapoints: x_batch.shape[0],
                }
            )

            total_nll += nll
            total_mse += mse

        total_nll /= n_tr
        total_mse /= n_tr

        for i in range(n_va):
            nll_val, mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val[log_batch * i:log_batch * (i + 1)],
                    self.Y_Minibatch: y_val[log_batch * i:log_batch * (i + 1)],
                    self.Weight_Minibatch: np.zeros((self.batch_size, self.n_nets)),
                    self.n_datapoints: X_val.shape[0],
                }
            )

            total_nll_val += nll_val
            total_mse_val += mse_val

        total_nll_val /= n_va
        total_mse_val /= n_va

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

    def train_normal(self, X, y, X_val=None, y_val=None, step_size=1e-3, max_iters=8000):
        start_time = time()

        """ Create optimizer """

        # Reinitialize adam
        logging.info("Reinitialize dynamics Adam")
        self.session.run(self.dynamics_adam_init)

        logging.info("Start Adam Training")

        if type(X) == tuple:
            generator = self.batch_generator(
                x=X[0], x_new=X[1], x_placeholder=self.X_Minibatch,
                y=y[0], y_new=y[1], y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                continual_placeholder=self.continual_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = np.vstack([X[0], X[1]])
            y_batch = np.vstack([y[0], y[1]])

        else:
            generator = self.batch_generator(
                x=X, x_placeholder=self.X_Minibatch,
                y=y, y_placeholder=self.Y_Minibatch,
                weight=self.log_weights, weight_placeholder=self.Weight_Minibatch,
                n_points_placeholder=self.n_datapoints,
                continual_placeholder=self.continual_train,
                batch_size=self.batch_size,
                seed=self.seed
            )

            x_batch = X
            y_batch = y

        for j in range(0, max_iters + 1):
            batch_dict = next(generator)

            _, training_loss = self.session.run([self._adam_op_opt, self.Nll], batch_dict)
            if j % 250 == 0:
                self.log_full_training_error(x_batch, y_batch, X_val, y_val,
                                             iteration_index=j, start_time=start_time, is_sampling=False)

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

        logging.info("Start sampling")

        # For logging
        if type(X) == tuple:
            x_batch = np.vstack([X[0], X[1]])
            y_batch = np.vstack([y[0], y[1]])
        else:
            x_batch = X
            y_batch = y

        logging_intervals = {"burn-in": 512, "sampling": sample_steps}

        sample_chain = itertools.islice(self.sampler, n_iters)

        for iteration_index, (parameter_values, _) in enumerate(sample_chain):

            burning_in = iteration_index <= burn_in_steps

            if burning_in and iteration_index % logging_intervals["burn-in"] == 0:

                self.log_full_training_error(x_batch, y_batch, X_val, y_val,
                                             iteration_index=iteration_index, start_time=start_time, is_sampling=False)

            if not burning_in and iteration_index % logging_intervals["sampling"] == 0:

                self.log_full_training_error(x_batch, y_batch, X_val, y_val,
                                             iteration_index=iteration_index, start_time=start_time, is_sampling=True)

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

        self.assign_pred_params.clear()
        i = 0

        for sample in self.samples:
            # TODO: Robust this !
            net_param = sample[2:]  # remove logQ and log_lambda

            for param_val in net_param:
                self.assign_pred_params.append(self.pred_network_params[i].assign(param_val))
                i += 1

        self.session.run(list(self.assign_pred_params))[0]  # For small value return

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

