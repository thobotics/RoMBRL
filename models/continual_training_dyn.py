# -*- coding: utf-8 -*-

"""
    continual_training_dyn.py
    
    Created on  : February 26, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
import numpy as np
import tensorflow as tf
from lib.utils.running_mean_std import RunningMeanStd
from models.continual_bnn_dyn import BayesNeuralNetDynModel
from models.continual_me_dyn import EnsembleNeuralNetDynModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TrainingDynamics(object):

    def __init__(self, n_inputs, n_outputs, n_timestep,
                 session=None, scope="training_dynamics", model_type="BNN",
                 policy=None, action_bounds=None,
                 batch_size=50, n_nets=100, n_units=50, activation=tf.tanh,
                 scale=1.0, a0=1.0, b0=0.1, a1=1.0, b1=0.1):

        assert model_type in ["BNN", "ME"]

        # Sanitize inputs
        assert isinstance(n_inputs, int)
        assert isinstance(n_outputs, int)
        assert isinstance(n_timestep, int)
        assert isinstance(batch_size, int)
        assert isinstance(n_nets, int)
        assert isinstance(n_units, int)

        assert n_inputs > 0
        assert n_outputs > 0
        assert n_timestep > 0
        assert batch_size > 0
        assert n_nets > 0
        assert n_units > 0

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.T = n_timestep
        self.batch_size = batch_size

        self.xu = np.array([], dtype=np.float32).reshape(0, n_inputs)
        self.y = np.array([], dtype=np.float32).reshape(0, n_outputs)

        # Setup model
        self.tf_scope = scope
        if session is not None:
            self.tf_sess = session
        else:
            self.tf_sess = tf.InteractiveSession()
        self.policy = policy
        self.action_bounds = action_bounds

        with tf.variable_scope("%s/input_rms" % self.tf_scope):
            self.input_rms = RunningMeanStd(epsilon=0.0, shape=self.n_inputs)
        with tf.variable_scope("%s/output_rms" % self.tf_scope):
            self.output_rms = RunningMeanStd(epsilon=0.0, shape=self.n_outputs)

        # Apply min max scaler on data for computational loss stabilization
        scale_range = (-10.0, 10.0)
        self.scaler_xu = MinMaxScaler(scale_range)
        self.scaler_y = MinMaxScaler(scale_range)
        # self.scaler_xu = StandardScaler(copy=True, with_mean=False, with_std=False)
        # self.scaler_y = StandardScaler(copy=True, with_mean=False, with_std=False)

        if model_type == "BNN":
            self.model = BayesNeuralNetDynModel(
                session=self.tf_sess, tf_scope=self.tf_scope,
                batch_size=self.batch_size, n_nets=n_nets,
                activation=activation, n_units=n_units,
                n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                scale=scale, a0=a0, b0=b0, a1=a1, b1=b1,
                normalize_input=self.input_rms, normalize_output=self.output_rms,
                dtype=tf.float32,
                # sampler arguments for SGHMC
                mdecay=0.05,
            )
        else:
            self.model = EnsembleNeuralNetDynModel(
                session=self.tf_sess, tf_scope=self.tf_scope,
                batch_size=self.batch_size, n_nets=n_nets,
                activation=activation, n_units=n_units,
                n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                scale=scale, a0=a0, b0=b0, a1=a1, b1=b1,
                normalize_input=self.input_rms, normalize_output=self.output_rms,
                dtype=tf.float32,
                # sampler arguments for SGHMC
                mdecay=0.05,
            )

    def _fit(self, x, u, y):
        xu = np.concatenate([x, u], axis=1)

        self.scaler_xu.partial_fit(xu)
        self.scaler_y.partial_fit(y)

        """ Update normalize parameters """
        self.input_rms.update(xu)
        self.output_rms.update(y - x)

    def add_data(self, x, u, y, method="append", fit=True):
        """ Add data to bundle

        Parameters
        ----------
         y : observation
         u : actions
         method: append or replace
        """
        assert method in ["append", "replace"]

        xu = np.concatenate([x, u], axis=1)

        if method == "append":
            self.xu = np.concatenate([self.xu, xu], axis=0)
            self.y = np.concatenate([self.y, y], axis=0)
        else:
            self.xu = xu
            self.y = y

            # TODO: Reinitialize input and output rms here !

        if fit:
            self._fit(x, u, y)

        return

    def _transform_data(self, x_val, y_val, x_tr_new=None, y_tr_new=None):

        x_tr_old = self.scaler_xu.transform(self.xu)
        y_tr_old = self.scaler_y.transform(self.y)
        x_val = self.scaler_xu.transform(x_val)
        y_val = self.scaler_y.transform(y_val)

        if x_tr_new is None:
            x_train = x_tr_old
            y_train = y_tr_old
        else:
            x_train = (x_tr_old, self.scaler_xu.transform(x_tr_new))
            y_train = (y_tr_old, self.scaler_y.transform(y_tr_new))

        return x_train, y_train, x_val, y_val

    def run_bnn(self, x_val=None, y_val=None, x_tr_new=None, y_tr_new=None,
                run_normal=True, normal_step_size=2.0e-3, normal_max_iters=3000,
                step_size=2.0e-3, mdecay=0.05, burn_in_steps=3000, n_iters=5000, sample_steps=100):

        assert isinstance(self.model, BayesNeuralNetDynModel)

        x_train, y_train, x_val, y_val = self._transform_data(x_val, y_val, x_tr_new, y_tr_new)

        if run_normal:
            self.model.train_normal(x_train, y_train, x_val, y_val,
                                    step_size=normal_step_size, max_iters=normal_max_iters)

        self.model.train(x_train, y_train, x_val, y_val,
                         step_size=step_size, mdecay=mdecay,
                         burn_in_steps=burn_in_steps, n_iters=n_iters, sample_steps=sample_steps)

    def run_normal(self, x_val=None, y_val=None, x_tr_new=None, y_tr_new=None, step_size=2.0e-3, max_iters=3000):

        x_train, y_train, x_val, y_val = self._transform_data(x_val, y_val, x_tr_new, y_tr_new)

        self.model.train_normal(x_train, y_train, x_val, y_val, step_size=step_size, max_iters=max_iters)

    def predict(self, x_test, **kwargs):
        x_test = self.scaler_xu.transform(x_test)

        var_test = None

        if "return_individual_predictions" in kwargs and not kwargs["return_individual_predictions"]:
            f_out = []

            for sample in self.model.samples:
                out = self.model.compute_network_output(params=sample, input_data=x_test)
                f_out.append(self.scaler_y.inverse_transform(out))
                # f_out.append(out)

            f_out = np.asarray(f_out)

            mu_test = np.mean(f_out, axis=0)
            var_test = np.mean((f_out - mu_test) ** 2, axis=0)

        else:
            mu_test, _ = self.model.predict(x_test, **kwargs)
            mu_test = self.scaler_y.inverse_transform(mu_test)

        return mu_test, var_test

    def _get_covariances(self):
        """ Fetch covariances from tensorgraph
        """

        log_Q = self.model.get_variable("log_Q").reshape(-1)
        log_lambda = self.model.get_variable("log_lambda")

        Q = np.diag(np.exp(log_Q))
        lambda_ = np.exp(log_lambda)

        return Q, lambda_

