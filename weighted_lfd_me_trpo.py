# -*- coding: utf-8 -*-

"""
    bnn_trpo.py
    
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

from lib.me_trpo.utils import *
from policy.nn_policy import NNPolicy
from lib.me_trpo.envs import  *
from lib.utils.rllab_env_rollout import IterativeData
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from models.weighted_training_dyn import TrainingDynamics


if __name__ == '__main__':
    log_dir = "/home/thobotics/rllab/data/local/swimmer/model_free_T_200/training_logs"

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

    # data_dir = "/home/thobotics/rllab/data/local/swimmer/dynamic_iter_100/training_logs"
    data_dir = "/home/thobotics/rllab/data/local/snake/snake_2019_03_13_10_15_16_0001/training_logs"
    xu_training, y_training = pickle.load(open("%s/data_integrated.pkl" % data_dir, 'rb'))
    xu_validate, y_validate = pickle.load(open("%s/data_val_integrated.pkl" % data_dir, 'rb'))

    # data_dir = "./swimmer_result/lfd_me_trpo/"
    data_dir = "./snake_result/weighted_me_trpo/"
    lfd_data_generator = IterativeData(n_states, n_actions, n_timestep, n_training=2000, n_validate=1000)
    lfd_data_generator.set_offline(xu_training, y_training, xu_validate, y_validate)

    data_generator = IterativeData(n_states, n_actions, n_timestep, n_training=2000, n_validate=1000)
    # data_generator.set_offline(xu_training, y_training, xu_validate, y_validate)

    """ TF session """
    sess = tf.InteractiveSession()

    """ Load dynamics """
    training = TrainingDynamics(n_inputs, n_outputs, n_timestep, action_bounds=env.action_space.bounds, session=sess,
                                model_type="ME", n_nets=10, n_units=1024, activation=tf.nn.relu,
                                batch_size=100, scale=1.0, a0=1.0, b0=1.0, a1=1.0, b1=5.0)

    """ Load policy """
    nn_policy = NNPolicy(sess, env, training, n_timestep, n_states, n_actions, log_dir=data_dir)

    # TODO: Clear this. Just for test the stability
    policy_dir = "/home/thobotics/Uncertainty_MBML/thobotics/BnnTRPO/snake_result/weighted_me_trpo/stuck_policy"
    # nn_policy.policy_saver.restore(sess, os.path.join(policy_dir, 'policy.ckpt'))
    xu_training, y_training, xu_validate, y_validate = pickle.load(open("%s/rollout.pkl" % policy_dir, 'rb'))
    data_generator.set_offline(xu_training, y_training, xu_validate, y_validate)
    training.add_data(xu_training[:, :n_states], xu_training[:, n_states:], y_training)

    """ Learning from demonstration """

    x_demo = np.array([]).reshape((0, n_states))
    u_demo = np.array([]).reshape((0, n_actions))
    y_demo = np.array([]).reshape((0, n_states))

    # for itr in range(8, 9):
    #
    #     logging.info("LfD Iteration %d" % itr)
    #
    #     """ Rollout and add data """
    #     x_tr, u_tr, y_tr, x_va, u_va, y_va = lfd_data_generator.fetch_data(itr)
    #     training.add_data(x_tr, u_tr, y_tr)
    #
    #     x_demo = np.concatenate([x_demo, x_tr], axis=0)
    #     u_demo = np.concatenate([u_demo, u_tr], axis=0)
    #     y_demo = np.concatenate([y_demo, y_tr], axis=0)
    #
    #     logging.info("Current training, validate: %d, %d" % (training.y.shape[0], y_va.shape[0]))
    #
    #     """ Optimize dynamics """
    #     xu_va = np.hstack((x_va, u_va))
    #     training.run_normal(np.hstack((x_va, u_va)), y_va, step_size=1.0e-3, max_iters=5000)
    #
    #     for i in range(training.model.n_nets):
    #         log_Q = training.model.get_variable("log_Q", model_idx=i).reshape(-1)
    #         log_lambda = training.model.get_variable("log_lambda", model_idx=i)
    #
    #         Q_new = np.diag(np.exp(log_Q))
    #         lambda_new = np.exp(log_lambda)
    #
    #         print(Q_new, "\n", "\n", lambda_new)
    #
    #     lfd_data_generator.plot_traj(training, iter=itr, n_sample=10,
    #                              data_path=os.path.join(data_dir, 'lfd_me_trpo'))
    #
    #     """ Optimize Policy """
    #     policy_costs = nn_policy.optimize_policy(100)
    #
    #     with open("%s/policy_log.txt" % data_dir, 'a') as f:
    #         f.write("Lfd iter %d %s\n" % (itr, " ".join(map(str, policy_costs))))

    """ RL step """

    start_itr = 10

    for itr in range(start_itr, start_itr+10):

        logging.info("Iteration %d" % itr)

        """ Rollout and add data """
        data_generator.rollout(nn_policy)
        x_tr, u_tr, y_tr, x_va, u_va, y_va = data_generator.fetch_data(itr)

        # Dump data
        pickle.dump((data_generator.xu_training, data_generator.y_training,
                     data_generator.xu_validate, data_generator.y_validate),
                    open("%s/rollout.pkl" % data_dir, 'wb'))

        if itr == start_itr:
            training.add_data(x_tr, u_tr, y_tr)
        else:
            training._fit(x_tr, u_tr, y_tr)

        logging.info("Current training, validate: %d, %d" % (training.y.shape[0], y_va.shape[0]))

        """ Optimize dynamics """
        xu_va = np.hstack((x_va, u_va))
        xu_new = np.vstack((np.hstack((x_tr, u_tr)), np.hstack((x_demo, u_demo))))
        y_new = np.vstack((y_tr, y_demo))
        training.run_normal(xu_va, y_va, xu_new, y_new, step_size=1e-3, max_iters=5000)

        for i in range(training.model.n_nets):
            log_Q = training.model.get_variable("log_Q", model_idx=i).reshape(-1)
            log_lambda = training.model.get_variable("log_lambda", model_idx=i)

            Q_new = np.diag(np.exp(log_Q))
            lambda_new = np.exp(log_lambda)

            print(Q_new, "\n", "\n", lambda_new)

        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(data_dir, 'me_trpo'))

        """ Optimize Policy """
        policy_costs = nn_policy.optimize_policy(100)

        with open("%s/policy_log.txt" % data_dir, 'a') as f:
            f.write("iter %d %s\n" % (itr, " ".join(map(str, policy_costs))))

        # Add data after training
        if itr > start_itr:
            training.add_data(x_tr, u_tr, y_tr, fit=False)

    for itr in range(start_itr, start_itr+10):
        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(data_dir, 'all_me_trpo'))
