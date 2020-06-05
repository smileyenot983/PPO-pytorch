from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.robots.pendula.inverted_double_pendulum import InvertedDoublePendulum
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np

class InvertedDoublePendulumSparseBulletEnv(BaseBulletEnv):
    def __init__(self):
        self.robot = InvertedDoublePendulum()
        BaseBulletEnv.__init__(self, self.robot)
        self.stateId = -1

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)
        r = BaseBulletEnv._reset(self)
        if self.stateId < 0:
            self.stateId = self._p.saveState()
        return r

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()  # sets self.pos_x self.pos_y
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
        dist_penalty = 0.01 * self.robot.pos_x ** 2 + (self.robot.pos_y + 0.3 - 2) ** 2
        # v1, v2 = self.model.data.qvel[1:3]   TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
        # vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        vel_penalty = 0
        alive_bonus = 10
        done = self.robot.pos_y + 0.3 <= 1
        #original reward:
        #self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
        #sparse reward(reward only for pendulums straight up position)
        #state consists of [x,vx,pos_x, cos(theta),sin(theta),theta_dot,  cos(gamma),sin(gamma),gamma_dot
        #in order to check position we can look at sin and cos of state
        

        #condition for pendulum 1 to be straight up
        #condition_1 = np.isclose([state[3],state[4]],[0,1],atol=0.01)
        #condition for pendulum 2 to be straight up
        #condition_2 = np.isclose([state[6],state[7]],[0,1],atol=0.01)
        # conditions = np.isclose([state[3],state[4],state[6],state[7]],[0,1,0,1],atol=0.01)
        
        #modified condition for pendulum 1:
        angle1 = np.arctan2(state[4], state[3])
        condition1 = np.pi/4 < angle1 < 3*np.pi/4
        angle2 = np.arctan2(state[6],state[7])
        condition2 = np.pi/4 < angle2 < 3*np.pi/4

        # reward = state[4] + state[7]

        if condition1 and condition2:
            reward = 1
        else:
            reward = 0
        self.HUD(state, a, done)
        

        return state, reward, done, {}

    def camera_adjust(self):
        self.camera.move_and_look_at(0,1.2,1.2, 0,0,0.5)
