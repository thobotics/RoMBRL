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

sys.path.append(os.path.abspath(os.path.join("lib", "pysgmcmc/")))

from lib.utils.misc import *
from policy.nn_policy import NNPolicy
from lib.utils.rllab_env_rollout import IterativeData
from lib.utils.env_helpers import get_env
from models.continual_training_dyn import TrainingDynamics

logger_lvl = logging.getLogger()
logger_lvl.setLevel(logging.DEBUG)


def run_main(params_dir, output_dir):

    """ Init data """
    all_params = load_params(params_dir)

    env = get_env(all_params["env"])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_inputs = env.observation_space.shape[0] + env.action_space.shape[0]
    n_outputs = env.observation_space.shape[0]
    n_timestep = all_params["n_timestep"]
    n_training = all_params["sample_train_size"]
    n_validate = all_params["sample_valid_size"]

    logging.info("Environment #States %d, #Actions %d", n_states, n_actions)

    """ Iterative data """

    data_generator = IterativeData(n_states, n_actions, n_timestep,
                                   n_training=n_training, n_validate=n_validate,
                                   rollout_params=all_params["rollout_params"])

    """ TF session """
    sess = tf.InteractiveSession()

    """ Load dynamics """
    training = TrainingDynamics(n_inputs, n_outputs, n_timestep,
                                action_bounds=env.action_space.bounds, session=sess,
                                model_type=all_params["model"],
                                dynamic_params=all_params["dynamics_params"],
                                dynamic_opt_params=all_params["dynamics_opt_params"])

    if all_params["restore_dir"]:
        logging.info("Restoring dynamics %s" % all_params["restore_dir"])
        xu_training, y_training, xu_validate, y_validate = pickle.load(open("%s/rollout.pkl"
                                                                            % all_params["restore_dir"], "rb"))
        training.add_data(xu_training[:, :n_states], xu_training[:, n_states:], y_training)
        training.restore(all_params["restore_dir"])

    """ Load policy """
    nn_policy = NNPolicy(sess, env, training, n_timestep, n_states, n_actions,
                         log_dir=output_dir,
                         policy_params=all_params["policy_params"],
                         policy_opt_params=all_params["policy_opt_params"])

    if all_params["restore_dir"]:
        nn_policy.policy_saver.restore(sess, os.path.join(all_params["restore_dir"], 'policy.ckpt'))

    """ RL step """

    start_itr, end_iter = 0, all_params["sweep_iters"]

    for itr in range(start_itr, end_iter):

        logging.info("Iteration %d" % itr)

        """ Rollout and add data """
        data_generator.rollout(nn_policy)
        x_tr, u_tr, y_tr, x_va, u_va, y_va = data_generator.fetch_data(itr)

        logging.info("Current training, validate: %d, %d" % (training.y.shape[0], y_va.shape[0]))

        """ Optimize dynamics """
        xu_va = np.hstack((x_va, u_va))

        training.add_data(x_tr, u_tr, y_tr)
        training.run(xu_va, y_va, params=all_params["dynamics_opt_params"])

        training.save(output_dir)
        logging.info("Saved dynamics %s" % output_dir)

        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(output_dir, 'bnn_trpo'))

        """ Optimize Policy """
        policy_costs = nn_policy.optimize_policy()

        with open("%s/policy_log.txt" % output_dir, 'a') as f:
            f.write("iter %d %s\n" % (itr, " ".join(map(str, policy_costs))))

        # TODO: Output as table

    for itr in range(start_itr, end_iter):
        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(output_dir, 'all_bnn_trpo'))


if __name__ == '__main__':

    env_name = "snake"
    root_folder = os.path.dirname(os.path.abspath(__file__))

    params_dir = "%s/params/params-%s.json" % (root_folder, env_name)
    output_dir = "%s/results/%s/non_continual_bnn_trpo" % (root_folder, env_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_main(params_dir, output_dir)

