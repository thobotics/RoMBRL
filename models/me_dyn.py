# -*- coding: utf-8 -*-

"""
    continual_me_dyn.py
    
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

from models.nnet.architecture import get_default_net
from lib.utils.misc import minimize_and_clip
from models.nnet.loss_function import TrainingLoss


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

        self.log_weights = np.zeros((self.batch_size, self.n_nets))

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
        self.Weight_Minibatch = tf.placeholder(shape=(None, self.n_nets),
                                               dtype=self.dtype,
                                               name="Weight_Minibatch")
        self.continual_train = tf.placeholder_with_default(False, shape=[], name="continual_train")
        self.n_datapoints = tf.placeholder(dtype=tf.int32, shape=[], name="n_datapoints")

        # setup params for covariances and neural network parameters

        # Diagonal covariance
        self.log_Q = []
        self.log_lambda = []
        self.f_output = []
        self.pred_output = []
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

                self.log_Q.append(tf.Variable(
                    np.log(self.b0) * tf.ones([1, n_outputs]), dtype=self.dtype,
                    name="log_Q",
                ))

                self.log_lambda.append(tf.Variable(
                    np.log(np.random.gamma(self.a1, self.b1)), dtype=self.dtype,
                    name="log_lambda",
                ))

                net_output = self.get_net(inputs=X_input, n_outputs=n_outputs,  n_units=n_units,
                                          activation=self.activation, seed=self.seed, dtype=self.dtype)

                # Unnormalize output
                if self.norm_output is not None:
                    net_output = zero_mean_unit_var_unnormalization(net_output,
                                                self.norm_output.mean, self.norm_output.std)

                f_output = self.X_Minibatch[:, :n_outputs] + net_output
                pred_output = f_output + \
                              tf.random_normal(shape=(tf.shape(f_output))) * tf.exp(self.log_Q[i])
                self.f_output.append(f_output)
                self.pred_output.append(f_output)
                # self.pred_output.append(pred_output)

                self.network_params.append(tf.trainable_variables(model_scope))

        # Initialize Tensorflow
        self.session.run(tf.variables_initializer(tf.global_variables(self.tf_scope)))

        """ Create optimizers """

        self.training_loss, self.Nll, self.Mse, self.log_likelihood = [], [], [], []

        # set up tensors for negative log likelihood and mean squared error
        for i in range(self.n_nets):

            self.training_loss.append(TrainingLoss(y_logit=self.f_output[i], y_true=self.Y_Minibatch,
                                                   log_Q=self.log_Q[i], log_lambda=self.log_lambda[i],
                                                   a0=self.a0, b0=self.b0, a1=self.a1, b1=self.b1,
                                                   network_params=self.network_params[i],
                                                   n_datapoints_placeholder=self.n_datapoints,
                                                   weight_placeholder=self.Weight_Minibatch,
                                                   continual_train=self.continual_train, continual_method="kl",
                                                   batch_size=self.batch_size, dtype=self.dtype))

            self.Nll.append(self.training_loss[i].negative_log_posterior())

            self.Mse.append(self.training_loss[i].mse())

            self.log_likelihood.append(self.training_loss[i].log_likehood())

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

    def _compute_weights(self, X, y):

        if type(X) == tuple:
            x_train = np.vstack([X[0], X[1]])
            y_train = np.vstack([y[0], y[1]])
        else:
            x_train = X
            y_train = y

        self.log_weights = np.zeros((x_train.shape[0], self.n_nets))

        for i in range(self.n_nets):
            params = [var.eval() for var in self.network_params[i]]

            feed_dict = dict(zip(self.network_params[i], params))
            feed_dict[self.X_Minibatch] = x_train
            feed_dict[self.Y_Minibatch] = y_train

            self.log_weights[:, i] = self.session.run(self.log_likelihood[i], feed_dict=feed_dict)

    def log_full_training_error(self, x_batch, y_batch, X_val, y_val,
                                start_time, iteration_index):

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

        total_nll, total_mse = [0.] * self.n_nets, [0.] * self.n_nets
        total_nll_val, total_mse_val = [0.] * self.n_nets, [0.] * self.n_nets

        for i in range(n_tr):
            nll, mse = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: x_batch[log_batch * i:log_batch * (i + 1)],
                    self.Y_Minibatch: y_batch[log_batch * i:log_batch * (i + 1)],
                    self.Weight_Minibatch: np.zeros((self.batch_size, self.n_nets)),
                    self.n_datapoints: x_batch.shape[0],
                }
            )

            total_nll = np.add(total_nll, nll)
            total_mse = np.add(total_mse, mse)

        total_nll = np.divide(total_nll, n_tr)
        total_mse = np.divide(total_mse, n_tr)

        for i in range(n_va):
            nll_val, mse_val = self.session.run(
                [self.Nll, self.Mse], feed_dict={
                    self.X_Minibatch: X_val[log_batch * i:log_batch * (i + 1)],
                    self.Y_Minibatch: y_val[log_batch * i:log_batch * (i + 1)],
                    self.Weight_Minibatch: np.zeros((self.batch_size, self.n_nets)),
                    self.n_datapoints: X_val.shape[0],
                }
            )

            total_nll_val = np.add(total_nll_val, nll_val)
            total_mse_val = np.add(total_mse_val, mse_val)

        total_nll_val = np.divide(total_nll_val, n_va)
        total_mse_val = np.divide(total_mse_val, n_va)

        seconds_elapsed = time() - start_time

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

        # Reinitialize adam
        logging.info("Reinitialize dynamics Adam")
        self.session.run(self.dynamics_adam_init)

        logging.info("Start Training")

        for j in range(0, max_iters + 1):

            for i in range(self.n_nets):
                batch_dict = next(generator)
                _, training_loss = self.session.run([self._adam_op_opt[i], self.Nll[i]], batch_dict)

            if j % 250 == 0:
                self.log_full_training_error(x_batch, y_batch, X_val, y_val,
                                             iteration_index=j, start_time=start_time)

        self.is_trained = True

        # Compute log likelihood and predict params of all models
        self._compute_weights(X, y)

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

            return self.session.run(self.pred_output[index], feed_dict={self.X_Minibatch: X_test}), None