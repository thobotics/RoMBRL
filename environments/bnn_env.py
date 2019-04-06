# -*- coding: utf-8 -*-

"""
    bnn_env.py
    
    Created on  : February 21, 2019
        Author  : thobotics
        Name    : Tai Hoang
"""
import tensorflow as tf
import numpy as np

from rllab.envs.base import Env
from rllab.envs.base import Step


class VecSimpleEnv(object):
    def __init__(self, env, n_envs, max_path_length):
        self.env = env
        self.n_envs = n_envs
        self.num_envs = n_envs
        self.states = np.zeros((self.n_envs, env.observation_space.shape[0]))
        self.ts = np.zeros((self.n_envs,))
        self.max_path_length = max_path_length
        self.cur_model_idx = 0

    def reset(self, dones=None):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        for i, done in enumerate(dones):
            if done:
                self.states[i] = self.env.reset()
                # self.cur_model_idx[i] = np.random.randint(self.env.bnn_model.model.n_nets)

        self.ts[dones] = 0
        self.cur_model_idx = (self.cur_model_idx + 1) % self.env.bnn_model.model.n_nets

        return self.states[dones]

    def step(self, actions):
        self.ts += 1
        actions = np.clip(actions, *self.env.action_space.bounds)
        next_observations = self.get_next_observation(actions)
        rewards = - self.env.cost_np(self.states, actions, next_observations)
        self.states = next_observations
        dones = self.env.is_done(self.states, next_observations)
        dones[self.ts >= self.max_path_length] = True
        if np.any(dones):
            self.reset(dones)

        return self.states, rewards, dones, dict()

    def get_next_observation(self, actions):
        next_observations, _ = self.env.bnn_model.predict(np.concatenate([self.states, actions], axis=1),
                                                          return_individual_predictions=True,
                                                          model_idx=self.cur_model_idx)

        return next_observations


class BayesNeuralNetEnv(Env):

    def __init__(self, env, inner_env, cost_np, bnn_model, sam_mode):
        self.vectorized = True
        self.env = env
        self.cost_np = cost_np
        self.is_done = getattr(inner_env, 'is_done', lambda x, y: np.asarray([False] * len(x)))
        self.bnn_model = bnn_model
        self.sam_mode = sam_mode
        super(BayesNeuralNetEnv, self).__init__()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        self._state = self.env.reset()
        observation = np.copy(self._state)

        return observation

    def step(self, action):
        action = np.clip(action, *self.action_space.bounds)
        next_observation, _ = self.bnn_model.predict(np.concatenate([self._state, action])[None])

        reward = - self.cost_np(self._state[None], action[None], next_observation)
        done = self.is_done(self._state[None], next_observation)[0]
        self._state = np.reshape(next_observation, -1)

        return Step(observation=self._state, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

    def vec_env_executor(self, n_envs, max_path_length):
        return VecSimpleEnv(env=self, n_envs=n_envs, max_path_length=max_path_length)