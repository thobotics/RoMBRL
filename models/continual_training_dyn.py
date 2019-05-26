# -*- coding: utf-8 -*-

"""
    continual_training_dyn.py
    
    Created on  : February 26, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
import logging
import pickle
import numpy as np
import tensorflow as tf
from lib.utils.running_mean_std import RunningMeanStd
from models.continual_bnn_dyn import BayesNeuralNetDynModel
from models.continual_me_dyn import EnsembleNeuralNetDynModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TrainingDynamics(object):

    def __init__(self, n_inputs, n_outputs, n_timestep,
                 session=None, scope="training_dynamics", model_type="BNN",
                 policy=None, action_bounds=None, dynamic_params=None, dynamic_opt_params=None):

        assert model_type in ["BNN", "ME"]

        n_nets = dynamic_params["n_nets"]
        n_units = dynamic_params["hidden_layers"]
        activation = dynamic_params["nonlinearity"]
        batch_size = dynamic_opt_params["batch_size"]
        scale = dynamic_opt_params["scale"]
        a0 = dynamic_opt_params["a0"]
        b0 = dynamic_opt_params["b0"]
        a1 = dynamic_opt_params["a1"]
        b1 = dynamic_opt_params["b1"]

        # Sanitize inputs
        assert isinstance(n_inputs, int)
        assert isinstance(n_outputs, int)
        assert isinstance(n_timestep, int)
        assert isinstance(batch_size, int)
        assert isinstance(n_nets, int)
        assert isinstance(n_units, list)
        assert isinstance(activation, list)

        assert n_inputs > 0
        assert n_outputs > 0
        assert n_timestep > 0
        assert batch_size > 0
        assert n_nets > 0

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

        self.model_saver = []
        self.has_data = False

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

            for i in range(self.model.n_nets):
                model_scope = "%s/model_%d" % (self.tf_scope, i)
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope)
                self.model_saver.append(tf.train.Saver(var_list))

    def fit(self, x, u, y):
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
            self.fit(x, u, y)

        self.has_data = True

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

    def run(self, x_val=None, y_val=None, x_tr_new=None, y_tr_new=None, params=None):

        if isinstance(self.model, BayesNeuralNetDynModel):
            self.run_bnn(x_val, y_val, x_tr_new, y_tr_new, params["bnn"])
        elif isinstance(self.model, EnsembleNeuralNetDynModel):
            self.run_normal(x_val, y_val, x_tr_new, y_tr_new, params["me"])

    def run_bnn(self, x_val=None, y_val=None, x_tr_new=None, y_tr_new=None, bnn_params=None):

        assert isinstance(self.model, BayesNeuralNetDynModel)

        run_normal = bnn_params["run_normal"]
        normal_step_size = bnn_params["normal_step_size"]
        normal_max_iters = bnn_params["normal_max_iters"]
        step_size = bnn_params["step_size"]
        mdecay = bnn_params["mdecay"]
        burn_in_steps = bnn_params["burn_in_steps"]
        max_passes = bnn_params["max_passes"]
        sample_steps = bnn_params["sample_steps"]

        x_train, y_train, x_val, y_val = self._transform_data(x_val, y_val, x_tr_new, y_tr_new)

        if run_normal:
            self.model.train_normal(x_train, y_train, x_val, y_val,
                                    step_size=normal_step_size, max_iters=normal_max_iters)

        self.model.train(x_train, y_train, x_val, y_val,
                         step_size=step_size, mdecay=mdecay,
                         burn_in_steps=burn_in_steps, n_iters=max_passes, sample_steps=sample_steps)

    def run_normal(self, x_val=None, y_val=None, x_tr_new=None, y_tr_new=None, me_params=None):

        max_iters = me_params["normal_max_iters"]
        step_size = me_params["step_size"]

        x_train, y_train, x_val, y_val = self._transform_data(x_val, y_val, x_tr_new, y_tr_new)

        self.model.train_normal(x_train, y_train, x_val, y_val, step_size=step_size, max_iters=max_iters)

    def predict(self, x_test, **kwargs):
        x_test = self.scaler_xu.transform(x_test)

        var_test = None

        if "return_individual_predictions" in kwargs and not kwargs["return_individual_predictions"]:
            f_out = []

            for sample in self.model.samples:
                out = self.model.compute_network_output(params=sample, input_data=x_test)
                out = np.clip(out, -100., 100.)
                f_out.append(self.scaler_y.inverse_transform(out))
                # f_out.append(out)

            f_out = np.asarray(f_out)

            mu_test = np.mean(f_out, axis=0)
            var_test = np.mean((f_out - mu_test) ** 2, axis=0)

        else:
            mu_test, _ = self.model.predict(x_test, **kwargs)
            mu_test = np.clip(mu_test, -100., 100.)
            mu_test = self.scaler_y.inverse_transform(mu_test)

        return mu_test, var_test

    def _log_covariances(self):
        """ Fetch covariances from tensorgraph
        """

        if isinstance(self.model, BayesNeuralNetDynModel):
            log_Q = self.model.get_variable("log_Q").reshape(-1)
            log_lambda = self.model.get_variable("log_lambda")

            Q = np.exp(log_Q)
            lambda_ = np.exp(log_lambda)

            logging.debug("Q:\t%s" % np.array2string(Q))
            logging.debug("Lambda:\t%s" % np.array2string(lambda_))
        else:
            for model_idx in range(self.model.n_nets):

                logging.debug("Model %d" % model_idx)

                log_Q = self.model.get_variable("log_Q", model_idx).reshape(-1)
                log_lambda = self.model.get_variable("log_lambda", model_idx)

                Q = np.exp(log_Q)
                lambda_ = np.exp(log_lambda)

                logging.debug("Q:\t%s" % np.array2string(Q))
                logging.debug("Lambda:\t%s" % np.array2string(lambda_))

    def save(self, save_dir):
        if isinstance(self.model, BayesNeuralNetDynModel):
            pickle.dump(self.model.samples, open("%s/bnn_dynamic_samples.pkl" % save_dir, 'wb'))
        else:
            for i in range(len(self.model_saver)):
                self.model_saver[i].save(self.tf_sess, "%s/me_%02d_dynamic_samples" % (save_dir, i),
                                         write_meta_graph=False)

    def restore(self, restore_dir):
        if isinstance(self.model, BayesNeuralNetDynModel):
            dynamic_samples = pickle.load(open("%s/bnn_dynamic_samples.pkl" % restore_dir, "rb"))
            self.model.samples.clear()
            for sample in dynamic_samples:
                self.model.samples.append(sample)

            assign_params = []
            for param, val in zip(self.model.network_params, dynamic_samples[-1]):
                assign_params.append(param.assign(val))
            self.tf_sess.run(assign_params)

            self.model.is_trained = True
            self.model._compute_weights(self.xu, self.y)
            self.model.feed_pred_params()
        else:
            for i in range(len(self.model_saver)):
                self.model_saver[i].restore(self.tf_sess, "%s/me_%02d_dynamic_samples" % (restore_dir, i))

            self.model.is_trained = True
            self.model._compute_weights(self.xu, self.y)

