# -*- coding: utf-8 -*-

"""
    diff_drive.py
    
    Created on  : April 06, 2019
        Author  : anonymous
        Name    : Anonymous
"""

from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import math
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
import tensorflow as tf


def diffdrive(dt, state, action):
    x, y, theta = state[:3]
    x_vel, y_vel, theta_vel = state[3:]
    v, w = action

    x_dot = v * math.cos(theta)
    y_dot = v * math.sin(theta)
    theta_dot = w

    next_state = np.array([x + dt * x_dot,
                           y + dt * y_dot,
                           theta + dt * theta_dot,
                           x_dot,
                           y_dot,
                           theta_dot])

    return next_state


class DiffDriveEnv(Env, Serializable):

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            n_timestep=200,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):

        self.dt = 0.05
        self.n_timestep = n_timestep
        self.boundary = np.array([-10, 10])
        self.vel_bounds = [-np.inf, np.inf]
        self.timestep = 0

        self.start_state = np.array([3.0, 0.0, np.pi, 0.0, 0.0, 0.0])
        self.goal_state = np.array([6.0, -6.0, 0.0, 0.0, 0.0, 0.0])
        # self.goal_state = np.array([6.0, 6.0, 0.0, 0.0, 0.0, 0.0])

        self.start_goal = (self.start_state, self.goal_state)

        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(DiffDriveEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def reset(self, start_state=None):
        self.timestep = 0
        start = np.copy(self.start_state)

        self._state = start
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        action = action if len(action.shape) == 1 else action[0]
        self._state = diffdrive(self.dt, self._state, action)

        goal_cost = np.linalg.norm(self.goal_state[:2] - self._state[:2])
        action_cost = self.ctrl_cost_coeff * np.mean(np.square(action), axis=0)

        if self.timestep == self.n_timestep - 1:
            goal_cost = 5000 * np.linalg.norm(self.goal_state[:3] - self._state[:3])
        cost = goal_cost + action_cost

        next_observation = np.copy(self._state)
        self.timestep += 1

        return Step(observation=next_observation, reward=-cost, done=False)

    def render(self):
        print('current state:', self._state)

    @property
    def observation_space(self):
        return Box(low=np.concatenate([self.boundary[0] * np.ones(3),
                                       self.vel_bounds[0] * np.ones(3)]),
                   high=np.concatenate([self.boundary[1] * np.ones(3),
                                        self.vel_bounds[1] * np.ones(3)]))

    @property
    def action_space(self):
        # return Box(low=-np.ones(2), high=np.ones(2))
        return Box(low=np.array([0.0, -1]),
                   high=np.array([1.0, 1]))

    def cost_rescale_action(self, action):
        lb, ub = self.action_space.bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        return np.clip(scaled_action, lb, ub)

    def cost_np(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        return np.mean(self.cost_np_vec(x, u, x_next))

    def cost_tf(self, x, u, x_next):
        u = self.cost_rescale_action(u)
        goal = self.goal_state

        goal_cost = tf.norm(goal[:2] - x_next[:, :2], axis=1)
        action_cost = self.ctrl_cost_coeff * tf.reduce_mean(tf.square(u), axis=1)
        if self.timestep == self.n_timestep - 1:
            goal_cost = 5000 * tf.norm(goal[:3] - x_next[:, :3], axis=1)

        return tf.reduce_mean(goal_cost + action_cost)

    def cost_np_vec(self, x, u, x_next):
        assert np.amax(np.abs(u)) <= 1.0
        u = self.cost_rescale_action(u)
        goal = self.goal_state

        goal_cost = np.linalg.norm(goal[:2] - x_next[:, :2], axis=1)
        action_cost = self.ctrl_cost_coeff * np.mean(np.square(u), axis=1)
        if self.timestep == self.n_timestep - 1:
            goal_cost = 5000 * np.linalg.norm(goal[:3] - x_next[:, :3], axis=1)

        return goal_cost + action_cost
