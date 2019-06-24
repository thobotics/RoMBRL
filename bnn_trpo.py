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
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "lib", "pysgmcmc")))

from lib.utils.misc import *
from lib.utils.rllab_env_rollout import IterativeData
from lib.utils.env_helpers import get_env
from models.training_dyn import TrainingDynamics
# import logging

logger_lvl = logging.getLogger()
logger_lvl.setLevel(logging.DEBUG)


def run_main(params_dir, output_dir, policy_type):

    """ Init data """
    all_params = load_params(params_dir)
    dump_params(all_params, output_dir)

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

    start_itr = 0
    if all_params["restore_dir"]:
        logging.info("Restoring dynamics %s" % all_params["restore_dir"])
        xu_training, y_training, xu_validate, y_validate = pickle.load(open("%s/rollout.pkl"
                                                                            % all_params["restore_dir"], "rb"))

        if "init_only" in all_params and all_params["init_only"]:
            init_lfd_traj = all_params["init_lfd_traj"]
            xu_training = xu_training[-init_lfd_traj * n_timestep:]
            y_training = y_training[-init_lfd_traj * n_timestep:]
            # xu_validate = xu_validate[:n_validate]
            # y_validate = y_validate[:n_validate]

        start_itr = len(xu_training) // n_training
        training.add_data(xu_training[:, :n_states], xu_training[:, n_states:], y_training)

        if "restore_dynamics" in all_params and all_params["restore_dynamics"]:
            training.restore(all_params["restore_dir"])
        else:
            training.run(xu_validate, y_validate, params=all_params["dynamics_opt_params"])

        data_generator.set_offline(xu_training, y_training, xu_validate, y_validate)

    """ Load policy """
    if policy_type == "lstm":
        from policy.nn_lstm_policy import NNPolicy
    else:
        from policy.nn_policy import NNPolicy

    nn_policy = NNPolicy(sess, env, training, n_timestep, n_states, n_actions,
                         log_dir=output_dir,
                         policy_params=all_params["policy_params"],
                         policy_opt_params=all_params["policy_opt_params"])

    if all_params["restore_dir"]:
        if not all_params["init_only"] and all_params["restore_policy"]:
            nn_policy.policy_saver.restore(sess, os.path.join(all_params["restore_dir"], 'policy.ckpt'))
        # else:
        #     nn_policy.optimize_policy()

    """ RL step """

    start_itr, end_iter = start_itr, all_params["sweep_iters"]

    # for itr in range(start_itr, start_itr+10):
    #     data_generator.rollout(nn_policy)
    #
    # start_itr += 10

    for itr in range(start_itr, end_iter):

        logging.info("Iteration %d" % itr)

        """ Rollout and add data """
        # if ("init_only" in all_params and all_params["init_only"] and itr > start_itr) or not all_params["init_only"]:
        #     data_generator.rollout(nn_policy)
        data_generator.rollout(nn_policy)
        x_tr, u_tr, y_tr, x_va, u_va, y_va = data_generator.fetch_data(itr)

        # Dump data
        pickle.dump((data_generator.xu_training, data_generator.y_training,
                     data_generator.xu_validate, data_generator.y_validate),
                    open("%s/rollout.pkl" % output_dir, 'wb'))

        logging.info("Current training, validate: %d, %d" % (training.y.shape[0], y_va.shape[0]))

        """ Optimize dynamics """
        xu_va = np.hstack((x_va, u_va))

        training.add_data(x_tr, u_tr, y_tr)
        training.run(xu_va, y_va, params=all_params["dynamics_opt_params"])

        training.save(output_dir)
        logging.info("Saved dynamics %s" % output_dir)

        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(output_dir, 'bnn_trpo'))

        training._log_covariances()

        """ Optimize Policy """
        est_costs, real_costs = nn_policy.optimize_policy()

        with open("%s/policy_log.txt" % output_dir, 'a') as f:
            f.write("iter %d\n"
                    "\t val [%s]\n"
                    "\t real [%s]\n"
                    "\t final real %.3f\n" % (itr,
                                              " ".join(map(str, est_costs)),
                                              " ".join(map(str, real_costs[:-1])),
                                              real_costs[-1]))

        data_generator.plot_fictitious_traj(training, nn_policy,
                                            data_path=os.path.join(output_dir, 'fict_samp_i%02d' % itr))

        # TODO: Output as table

    for itr in range(start_itr, end_iter):
        data_generator.plot_traj(training, iter=itr, n_sample=10,
                                 data_path=os.path.join(output_dir, 'all_bnn_trpo'))


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--indir',
                        required=False,
                        help="Params dir")

    parser.add_argument('-o', '--outdir',
                        required=False,
                        help="Output dir")

    parser.add_argument('-l', '--logdir',
                        required=False,
                        help="Log dir")

    parser.add_argument('-e', '--env',
                        required=False,
                        help="Environment")

    parser.add_argument('-g', '--gpu',
                        required=False,
                        help="Environment")

    parser.add_argument('-p', '--policy',
                        required=False,
                        help="Environment")

    args = parser.parse_args()

    params_dir = args.indir
    log_dir = args.logdir
    output_dir = args.outdir
    env_name = args.env
    gpu = args.gpu
    policy = args.policy

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    logging.basicConfig(filename=log_dir,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_main(params_dir, output_dir, policy)


if __name__ == "__main__":
    main(sys.argv[1:])

