# -*- coding: utf-8 -*-

"""
    nn_policy.py
    
    Created on  : February 28, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
import tensorflow as tf
import numpy as np
import sys
import os
import logging

import rllab.misc.logger as rllab_logger
from sandbox.rocky.tf.envs.base import TfEnv

# sys.path.append(os.path.abspath(os.path.join("lib", "me_trpo")))

from environments.bnn_env import BayesNeuralNetEnv
from rllab_algos.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from lib.utils.env_helpers import evaluate_fixed_init_trajectories
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy


"""
Extend from project ME-TRPO
"""
class NNPolicy(object):

    def __init__(self, session, env, dyn_model, n_timestep, n_states, n_actions, log_dir,
                 n_envs=10, min_iter=25, reset_non_increasing=3):
        self.env = env
        self.tf_sess = session
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_timestep = n_timestep
        self.scope_name = "training_policy"
        self.policy_saver = None
        self.log_dir = log_dir

        # Parameters assign
        self.n_envs = n_envs
        self.min_iters = min_iter
        self.reset_non_increasing = reset_non_increasing

        # Initial value
        self.min_validation_cost = 9999.
        self.non_increase_counter = 0

        self.training_policy, self.policy_model = self.build_policy_from_rllab(env=env, n_actions=self.n_actions)
        self.policy_in, self.policy_out = self.initialize_policy(self.policy_model, self.n_states)
        self.algo_policy, self.cost_np_vec = self.init_bnn_trpo(dyn_model, self.training_policy, self.n_timestep)

        return

    def build_policy_from_rllab(self, env, n_actions):
        """ Return both rllab policy and policy model function. """

        sess = self.tf_sess
        scope_name = self.scope_name

        # Initialize training_policy to copy from policy

        training_policy = GaussianMLPPolicy(
            name=scope_name,
            env_spec=env.spec,
            hidden_sizes=[32, 32],
            init_std=1.0,
            output_nonlinearity=tf.identity
        )
        training_policy_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='training_policy')
        sess.run([tf.variables_initializer(training_policy_vars)])

        # Compute policy model function using the same weights.
        training_layers = training_policy._mean_network.layers

        def policy_model(x, stochastic=0.0, collect_summary=False):
            assert (training_layers[0].shape[1] == x.shape[1])
            h = x

            for i, layer in enumerate(training_layers[1:]):
                w = layer.W
                b = layer.b
                pre_h = tf.matmul(h, w) + b
                h = layer.nonlinearity(pre_h, name='policy_out')

            std = training_policy._l_std_param.param
            h += stochastic * tf.random_normal(shape=(tf.shape(x)[0], n_actions)) * tf.exp(std)

            return h

        return training_policy, policy_model

    def initialize_policy(self, policy_model, n_states):

        # Initial tf variables
        policy_scope = self.scope_name
        policy_in = tf.placeholder(tf.float32, shape=(None, n_states), name='policy_in')
        policy_out = policy_model(policy_in)
        tf.add_to_collection("policy_in", policy_in)
        tf.add_to_collection("policy_out", policy_out)

        """ Prepare variables and data for learning """

        # Initialize all variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=policy_scope)
        init_op = tf.initialize_variables(var_list)
        self.tf_sess.run(init_op)

        # Policy saver
        self.policy_saver = tf.train.Saver(var_list)

        return policy_in, policy_out

    def get_action(self, observation, action_noise, **kwargs):

        if len(observation.shape) == 1:
            observation = observation[np.newaxis]

        action = self.tf_sess.run(self.policy_out, feed_dict={self.policy_in: observation})

        # More noisy as t increases, max_var = 1.0
        n_particles, n_actions = action.shape
        action += action_noise * np.random.randn(n_particles, n_actions)

        return np.clip(action, *kwargs['action_bounds'])

    def init_bnn_trpo(self, bnn_model, training_policy, time_step):

        if hasattr(self.env._wrapped_env, '_wrapped_env'):
            inner_env = self.env._wrapped_env._wrapped_env
        else:
            inner_env = self.env._wrapped_env.env.unwrapped

        cost_np_vec = inner_env.cost_np_vec

        bnn_env = TfEnv(BayesNeuralNetEnv(env=self.env,
                                          inner_env=inner_env,
                                          cost_np=cost_np_vec,
                                          bnn_model=bnn_model,
                                          sam_mode=None))

        baseline = LinearFeatureBaseline(env_spec=self.env.spec)

        sampler_args = {"n_envs": self.n_envs}
        batch_size = int(self.n_timestep * bnn_model.model.n_nets * sampler_args["n_envs"])

        algo = TRPO(
            env=bnn_env,
            policy=training_policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=time_step,
            discount=1.00,
            step_size=0.01,
            sampler_args=sampler_args,  # params for VectorizedSampler
        )

        return algo, cost_np_vec

    def optimize_policy(self, iteration):

        cost_np_vec = self.cost_np_vec
        algo = self.algo_policy
        real_env = self.env

        """ Re-initialize Policy std parameters. """
        if self.non_increase_counter == self.reset_non_increasing:
            self.tf_sess.run(tf.variables_initializer(tf.global_variables(self.scope_name)))
            self.non_increase_counter = 0
            self.min_validation_cost = 999.

        reset_op = tf.assign(self.training_policy._l_std_param.param, np.log(1.0) * np.ones(self.n_actions))
        print("Before reset policy std ", np.exp(self.training_policy._l_std_param.param.eval()))
        self.tf_sess.run([reset_op])

        """ Optimize policy via rllab. """

        min_iter = self.min_iters
        min_validation_cost = 9999.0  # self.min_validation_cost
        min_idx = 0
        policy_opt_batch_size = 100  # policy_opt_params.batch_size
        policy_validation_init = [real_env.reset() for i in range(policy_opt_batch_size)]
        policy_validation_reset_init = np.array(policy_validation_init)
        real_validation_costs = []

        for j in range(iteration):
            algo.start_worker()

            with rllab_logger.prefix('itr #%d | ' % int(j + 1)):
                paths = algo.obtain_samples(j)
                samples_data = algo.process_samples(j, paths)
                algo.optimize_policy(j, samples_data)

            """ Do validation cost """

            if (j + 1) % 5 == 0:

                real_validation_cost = evaluate_fixed_init_trajectories(
                    real_env,
                    self.policy_in,
                    self.policy_out,
                    policy_validation_reset_init,
                    cost_np_vec, self.tf_sess,
                    max_timestep=self.n_timestep,
                    gamma=1.00
                )
                real_validation_costs.append(real_validation_cost)

                logging.info('iter %d' % j)
                logging.info("\tValidation cost:\t%.3f", real_validation_cost)

                """ Store current best policy """
                if real_validation_cost < min_validation_cost:
                    min_idx = j
                    min_validation_cost = real_validation_cost

                    # Save
                    logging.info('\tSaving policy')
                    self.policy_saver.save(self.tf_sess,
                                           os.path.join(self.log_dir, 'policy.ckpt'),
                                           write_meta_graph=False)

                if j > min_iter and real_validation_cost - min_validation_cost > 1.0:  # tolerance
                    logging.info("Stop at iteration %d and restore the current best at %d: %.3f"
                                 % (j + 1, min_idx + 1, min_validation_cost))
                    self.policy_saver.restore(self.tf_sess, os.path.join(self.log_dir, 'policy.ckpt'))
                    break

        if min_validation_cost < self.min_validation_cost:
            self.min_validation_cost = min_validation_cost
            self.non_increase_counter = 0
        else:
            self.non_increase_counter += 1

        logging.info("Best in all iters %.3f, non increasing in %d" %
                     (self.min_validation_cost, self.non_increase_counter))

        return real_validation_costs
