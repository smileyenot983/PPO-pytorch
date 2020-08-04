"""
Cart pole swing-up: modified version of:
https://github.com/hardmaru/estool/blob/master/custom_envs/cartpole_swingup.py
"""


import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import rendering

class Pendubot(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #parameters of system
        self.l_1 = 0.25
        self.l_2 = 0.25
        self.m_1 = 1.085
        self.m_2 = 0.26
        self.l_c1 = self.l_1/2
        self.l_c2 = self.l_2/2
        self.I_1 = 1/3 * self.m_1 * self.l_1**2
        self.I_2 = 1/3 * self.m_2 * self.l_2**2
        self.gravity = 9.8

        #timestep duration
        self.dt = 0.001

        #maximum possible torque
        self.max_torque = 1.5

        high_obs = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        low_obs = -high_obs


        self.action_space = spaces.Box(low=-self.max_torque,
                                       high=self.max_torque,
                                       shape=(1,))

        self.observation_space = spaces.Box(low=low_obs,
                                            high=high_obs)


        self.state = None
        self.viewer = None

    def reset(self):
        self.state = np.array([-np.pi/2, 0.0, 0.0, 0.0]).reshape(-1,)
        return self.state


    def step(self, action):

        #integrating different terms into a variable to improve readability
        theta_1 = self.m_1 * self.l_c1 * self.l_c1 + self.m_2 * self.l_1**2 + self.I_1
        theta_2 = self.m_2 * self.l_c2**2 + self.I_2
        theta_3 = self.m_2 * self.l_1 * self.l_c2
        theta_4 = self.m_1 * self.l_c1 + self.m_2 * self.l_1
        theta_5 = self.m_2 * self.l_c2

        q_1,q_2,dq_1,dq_2 = self.state


        #Euler's integration to update states
        ddq_1 = 1/(theta_1*theta_2 - theta_3**2*np.cos(q_2)**2) * (theta_2*theta_3*np.sin(q_2)*(dq_1+dq_2)**2 + theta_3**2*np.cos(q_2)*np.sin(q_2)*dq_1**2 - theta_2*theta_4*self.gravity*np.cos(q_1) \
                                                                     + theta_3*theta_5*self.gravity*np.cos(q_2)*np.cos(q_1+q_2) + theta_2*action)

        ddq_2 = 1/(theta_1*theta_2 - theta_3**2 * np.cos(q_2)**2) * (-theta_3*(theta_2+theta_3*np.cos(q_2))*np.sin(q_2)*(dq_1+dq_2)**2 - (theta_1 + theta_3*np.cos(q_2))*theta_3 \
                                                                       *np.sin(q_2)*dq_1**2 + (theta_2+theta_3*np.cos(q_2))*(theta_4*self.gravity*np.cos(q_1) - action) - (theta_1+theta_3*np.cos(q_2)) \
                                                                       *theta_5*self.gravity*np.cos(q_1+q_2))

        dq_1 = dq_1 + ddq_1 * self.dt
        dq_2 = dq_2 + ddq_2 * self.dt


        q_1 = (q_1 + dq_1 * self.dt)
        q_2 = (q_2 + dq_2 * self.dt)


        self.state = np.array([q_1,q_2,dq_1,dq_2]).reshape(-1,)

        #calculating reward
        reward = np.sin(q_1)
        # reward = np.sin(q_1) + np.cos(q_2)
        # x_required = 0
        # y_required = self.l_1+self.l_2
        #
        # x_current = self.l_1*np.cos(q_1) + self.l_2*np.cos(q_1+q_2)
        # y_current = self.l_1*np.sin(q_1) + self.l_2*np.sin(q_1+q_2)

        # reward = -((x_current-x_required)**2 + (y_current-y_required)**2)

        #no conditions for ending the episode
        done = False
        # state_ret =
        return np.array([np.sin(q_1),np.cos(q_1),np.sin(q_2),np.cos(q_2)]).reshape(-1,), reward, done, {}

    def render(self, mode='human'):
        # global viewer

        q_1,q_2,dq_1,dq_2 = self.state
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.l_1 + self.l_2 + 0.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        # p1 = [-l_1[None] * np.cos(q_1[t]), l_1[None] * np.sin(q_1[t])]
        p1 = [self.l_1 * np.cos(q_1), self.l_1 * np.sin(q_1)]

        p2 = [p1[0] - self.l_2 * np.cos(q_1 + q_2),
              p1[1] + self.l_2 * np.sin(q_1 + q_2)]

        xys = np.array([[0, 0], p1, p2])  # [:,::-1]

        thetas = [q_1, q_1+q_2]
        # thetas = [q_1[t] - np.pi/2, q_1[t] + q_2[t] - np.pi/2]
        link_lengths = [self.l_1, self.l_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')