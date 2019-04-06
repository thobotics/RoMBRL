# -*- coding: utf-8 -*-

"""
    continual_bnn_trpo.py
    
    Created on  : February 28, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import tensorflow as tf
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join("lib", "me_trpo")))
sys.path.append(os.path.abspath(os.path.join("lib", "pysgmcmc/")))

from lib.utils.misc import *
from policy.nn_policy import NNPolicy
from environments.mujoco_envs import *
from lib.utils.rllab_env_rollout import IterativeData
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from models.continual_training_dyn import TrainingDynamics
from pympler import classtracker

if __name__ == '__main__':

    logger_lvl = logging.getLogger()
    logger_lvl.setLevel(logging.DEBUG)

    # env = TfEnv(normalize(SwimmerEnv()))
    env = TfEnv(normalize(SnakeEnv()))
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_inputs = env.observation_space.shape[0] + env.action_space.shape[0]
    n_outputs = env.observation_space.shape[0]
    n_timestep = 200  # 1000
    action_noise = 3.0

    logging.info("Environment #States %d, #Actions %d", n_states, n_actions)

    """ Iterative data """

    # data_dir = "./swimmer_result/lfd_bnn_trpo/"
    data_dir = "./results/snake/weighted_lfd_bnn_trpo/"
    data_generator = IterativeData(n_states, n_actions, n_timestep, n_training=1000, n_validate=1000)

    """ TF session """
    sess = tf.InteractiveSession()

    """ Load dynamics """
    training = TrainingDynamics(n_inputs, n_outputs, n_timestep, action_bounds=env.action_space.bounds, session=sess,
                                model_type="BNN", n_nets=20, n_units=1024, activation=tf.tanh,
                                batch_size=100, scale=1.0, a0=1.0, b0=1.0, a1=1.0, b1=5.0)

    """ Load policy """
    nn_policy = NNPolicy(sess, env, training, n_timestep, n_states, n_actions, log_dir=data_dir)

    """ RL step """

    start_itr, end_iter = 0, 30
    burn_in = [3000] * end_iter

    tracker = classtracker.ClassTracker()
    tracker.track_object(training)
    tracker.track_object(training.model)
    tracker.track_object(nn_policy)
    tracker.track_object(data_generator)
    tracker.create_snapshot()

    for itr in range(start_itr, end_iter):

        logging.info("Iteration %d" % itr)

        """ Rollout and add data """
        data_generator.rollout(nn_policy)
        x_tr, u_tr, y_tr, x_va, u_va, y_va = data_generator.fetch_data(itr)

        if itr == start_itr:
            training.add_data(x_tr, u_tr, y_tr)
        else:
            training._fit(x_tr, u_tr, y_tr)

        logging.info("Current training, validate: %d, %d" % (training.y.shape[0], y_va.shape[0]))

        """ Optimize dynamics """
        xu_va = np.hstack((x_va, u_va))
        xu_new = np.hstack((x_tr, u_tr))
        y_new = y_tr

        if itr == start_itr:
            training.run_bnn(xu_va, y_va,
                             run_normal=True, normal_max_iters=3000,
                             step_size=10.0e-3, mdecay=0.05,
                             burn_in_steps=5000, n_iters=100000, sample_steps=200)
        else:
            training.run_bnn(xu_va, y_va, xu_new, y_new,
                             run_normal=True, normal_max_iters=3000,
                             step_size=10.0e-3, mdecay=0.05,
                             burn_in_steps=burn_in[itr], n_iters=100000, sample_steps=200)

        Q_new, lambda_new = training._get_covariances()
        Q_new_str = np.array2string(Q_new, formatter={'float_kind': '{0:.5f}'.format})
        logging.debug("Q_new = %s \n lambda_new = %.3f" % (Q_new_str, lambda_new))

        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(data_dir, 'bnn_trpo'))

        """ Optimize Policy """
        policy_costs = nn_policy.optimize_policy(100)

        with open("%s/policy_log.txt" % data_dir, 'a') as f:
            f.write("iter %d %s\n" % (itr, " ".join(map(str, policy_costs))))

        # Add data after training
        if itr > start_itr:
            training.add_data(x_tr, u_tr, y_tr, fit=False)

        tracker.create_snapshot()
        tracker.stats.print_summary()

    for itr in range(start_itr, end_iter):
        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(data_dir, 'all_bnn_trpo'))

    tracker.stats.print_summary()

