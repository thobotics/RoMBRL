# -*- coding: utf-8 -*-

"""
    rllab_env_rollout.py
    
    Created on  : March 06, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""

import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "lib", "me_trpo")))

import logging
import rllab.misc.logger as rllab_logger
from lib.utils.namedtuples import Rollout_params
import matplotlib.pyplot as plt


# Sample a batch of trajectories from an environment
# Use tensorflow policy given as (in and out).
# Batch size is the total number of transitions (not trajectories).
def sample_trajectories(nn_policy, batch_size, exploration, render_every=None):

    env = nn_policy.env
    max_timestep = nn_policy.n_timestep
    cost_np = nn_policy.cost_np_vec

    Os = []
    As = []
    Rs = []
    max_eps_reward = -np.inf
    min_eps_reward = np.inf
    avg_eps_reward = 0.0
    _counter = 1

    while _counter <= batch_size:
        o = []
        a = []
        r = []

        observation = env.reset()
        o.append(observation)
        episode_reward = 0.0

        for t in range(max_timestep):
            # Perturb policy.
            if exploration['vary_trajectory_noise']:
                action_noise = exploration['action_noise']*np.random.uniform()
            else:
                action_noise = exploration['action_noise']

            action = nn_policy.get_action(observation,
                                          action_noise=action_noise,
                                          action_bounds=env.action_space.bounds)

            observation, reward, done, info = env.step(action)

            # # Debug is_done
            # if is_done is not None:
            #     assert done == is_done(o[-1][None], observation[None])[0]

            o.append(observation)
            a.append(action[0])
            r.append(reward)
            episode_reward += reward
            _counter += 1

            if render_every is not None and len(Os) % render_every == 0:
                env.render()
            if done:
                break

        # debugging cost function
        # if cost_np is not None:
        #     episode_cost = len(a) * cost_np(np.array(o[:-1]),
        #                                     np.array(a),
        #                                     np.array(o[1:]))
        #     # Check if cost_np + env_reward == 0
        #     logging.info('%d steps, cost %.2f, verify_cost %.3f'
        #                 % (_counter - 1,
        #                    episode_cost,
        #                    episode_reward + episode_cost))
        # else:
        #     logging.info('%d steps, reward %.2f'
        #                 % (_counter - 1, episode_reward))

        # Recover policy
        # saver.restore(sess, os.path.join(log_dir, 'policy.ckpt'))
        # logger.debug("Restored the policy back to %s" % os.path.join(log_dir, 'policy.ckpt'))

        Os.append(o)
        As.append(a)
        Rs.append(r)
        # Update stats
        avg_eps_reward += episode_reward
        if episode_reward > max_eps_reward:
            max_eps_reward = episode_reward
        if episode_reward < min_eps_reward:
            min_eps_reward = episode_reward

    avg_eps_reward /= len(Os)
    rllab_logger.record_tabular('EpisodesSoFar', len(Os))
    rllab_logger.record_tabular('TimeStepsSoFar', _counter - 1)
    return Os, As, Rs, {'avg_eps_reward': avg_eps_reward,
                        'min_eps_reward': min_eps_reward,
                        'max_eps_reward': max_eps_reward}


class IterativeData(object):
    def __init__(self, n_states, n_actions, n_timestep, n_training=2000, n_validate=1000):

        self.xu_training = np.array([], dtype=np.float32).reshape(0, n_states + n_actions)
        self.y_training = np.array([], dtype=np.float32).reshape(0, n_states)
        self.xu_validate = np.array([], dtype=np.float32).reshape(0, n_states + n_actions)
        self.y_validate = np.array([], dtype=np.float32).reshape(0, n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_timestep = n_timestep
        self.n_training = n_training
        self.n_validate = n_validate

        # TODO: Beautifier this
        rollout_params = {
            "training_data_size": 100000,
            "validation_data_size": 50000,
            "split_ratio": 0.33333333,
            "splitting_mode": "trajectory",
            "use_same_dataset": True,
            "exploration": {
                "initial_param_std": 0.0,
                "param_noise": 3.0,
                "action_noise": 3.0,
                "vary_trajectory_noise": True
            },
            "datapath": "",
            "is_monitored": False,
            "monitorpath": None,
            "max_timestep": n_timestep,
            "render_every": None,
            "load_rollout_data": False
        }

        self.rollout_params = Rollout_params(**rollout_params)

    def set_offline(self, xu_training, y_training, xu_validate, y_validate):

        self.xu_training = xu_training
        self.y_training = y_training
        self.xu_validate = xu_validate
        self.y_validate = y_validate

    def rollout(self, nn_policy):

        sample_size = self.n_training + self.n_validate

        """ Do rollout """
        Os, As, Rs, info = sample_trajectories(nn_policy, sample_size, self.rollout_params.exploration)

        """ Generate data """
        x_all = []
        y_all = []

        for i, o in enumerate(Os):
            a = As[i]

            for t in range(len(o) - 1):
                x_all.append(np.concatenate([o[t], a[t]]))
                y_all.append(o[t + 1])

        x_all = np.array(x_all)
        y_all = np.array(y_all)

        self.xu_training = np.concatenate([self.xu_training, x_all[:self.n_training]], axis=0)
        self.y_training = np.concatenate([self.y_training, y_all[:self.n_training]], axis=0)
        self.xu_validate = np.concatenate([self.xu_validate,
                                           x_all[self.n_training:self.n_training + self.n_validate]], axis=0)
        self.y_validate = np.concatenate([self.y_validate,
                                          y_all[self.n_training:self.n_training + self.n_validate]], axis=0)

    def fetch_data(self, iter, all_val=True):

        n_states = self.n_states

        idx_tr = slice(self.n_training * iter, self.n_training * (iter + 1))

        if all_val:
            idx_val = slice(0, self.xu_validate.shape[0])
        else:
            idx_val = slice(self.n_validate * iter, self.n_validate * (iter + 1))

        return self.xu_training[idx_tr, :n_states], self.xu_training[idx_tr, n_states:], self.y_training[idx_tr], \
               self.xu_validate[idx_val, :n_states], self.xu_validate[idx_val, n_states:], self.y_validate[idx_val]

    def plot_traj(self, bnn_model, iter, n_sample, data_path=None):

        n_sample = min(n_sample, self.n_training // self.n_timestep)
        # n_sample = min(n_sample, self.n_validate // self.n_timestep)
        n_states = self.n_states

        idx_tr = slice(self.n_training * iter, self.n_training * (iter + 1))
        # idx_tr = slice(self.n_validate * iter, self.n_validate * (iter + 1))

        color = ['green', 'k', 'yellow', 'cyan', "blue"]

        # Setup state for plotting
        for sample in range(n_sample):
            idx_traj = slice(self.n_timestep * sample, self.n_training * (sample + 1))
            # idx_traj = slice(self.n_timestep * sample, self.n_validate * (sample + 1))

            start_state = self.xu_training[idx_tr, :n_states][idx_traj][0]

            x_nn_test = np.zeros((n_states, self.n_timestep))
            x_nn_test[:, 0] = start_state

            set_actions = self.xu_training[idx_tr, n_states:][idx_traj]
            # set_actions = self.xu_validate[idx_tr, n_states:][idx_traj]

            # Plot all trajectories
            color_idx = 0

            plt.rcParams['figure.figsize'] = (8, 3)
            fig = plt.figure()

            # plt.xlim([-2.0, 10.0])
            # plt.ylim([-1.5, 1.5])

            plt.xlim([-2.0, 2.0])
            plt.ylim([-1.5, 1.5])

            # plt.xlim([-0.5, 2.0])
            # plt.ylim([-0.5, 0.5])

            for idx in range(bnn_model.model.n_nets):

                for i in range(self.n_timestep - 1):
                    x_tmp = np.concatenate([x_nn_test[:, i], set_actions.T[:, i] + np.random.randn(self.n_actions) * 0.0], axis=0)
                    x_new, _ = bnn_model.predict(x_tmp[:, np.newaxis].T,
                                                 return_individual_predictions=True, model_idx=idx)
                    x_nn_test[:, i + 1] = x_new.reshape(-1)

                trajectories = x_nn_test.T

                plt.quiver(trajectories[:, 0], trajectories[:, 1],
                           np.cos(trajectories[:, 2]), np.sin(trajectories[:, 2]), width=0.002,
                           color=color[color_idx], alpha=max(0.3, 1 - idx*0.1))
                color_idx = (color_idx + 1) % len(color)

            if data_path is None:
                plt.grid()
                plt.show()
            else:
                fig.savefig("%s_%02d_%02d.jpg" % (data_path, iter, sample))
                logging.debug("Saved trajectory %02d" % sample)

            plt.close()

