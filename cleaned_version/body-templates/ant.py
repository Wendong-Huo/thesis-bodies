import numpy as np

import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

from stable_baselines3.common import logger

from pybullet_envs.env_bases import Camera

from utils import output

class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, xml, param, name, env, powercoeffs=[1., 1., 1.]):
        self.step_num = 0
        self.env = env
        self.xml = xml
        # remove private params
        private_keys = []
        for key in param:
            if key.startswith("_"):
                private_keys.append(key)
        for key in private_keys:
            del param[key]
        self.param = param
        self.name = name
        self.powercoeffs = powercoeffs
        self.sqrt2 = np.sqrt(2)
        WalkerBase.__init__(self, xml, "torso", action_dim=8, obs_dim=28, power=2.5) # larger power? because we have larger bodies in variation. original = 2.5

    def alive_bonus(self, z, pitch):
        if z> self.param["size_torso"] + self.param["length_leg"]*self.sqrt2*2 + self.param["length_foot"]*self.sqrt2 and z>self.initial_z+0.1:
            # explicitly punish fly-away bug
            output("fly away bug.", -1)
            ret = -1e7
        else:
            ret = +1 if z > self.param["size_torso"]+0.1 else -1  # self.param["size_torso"] is central sphere rad, die if it scrapes the ground
        return ret


class AntEnv(WalkerBaseBulletEnv):

    def __init__(self, xml, param, name=None, powercoeffs=[1., 1., 1.], is_eval=False, render=False, max_episode_steps=1e3):
        self.robot = Ant(xml, param, name, env=self, powercoeffs=powercoeffs)
        self.is_eval = is_eval
        self.max_episode_steps = max_episode_steps
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.camera = CameraOnTop(self)
        self.electricity_cost /= 10.

    def reset(self):
        self.robot.step_num = 0
        obs = super().reset()
        return obs

    def step(self, a):
        if self.robot.step_num % 1000 == 0:
            for n in range(len(a)):
                logger.record(f"debug/motor_{n}", a[n])
        self.robot.step_num += 1
        obs, r, done, info = self.super_step(a)
        if self.robot.step_num > self.max_episode_steps:
            done = True
        if done:
            if self.is_eval:
                logger.record(f"eval/e{self.robot.name}_body_x", self.robot.body_xyz[0])
            else:
                logger.record(f"debug/d{self.robot.name}_body_x", self.robot.body_xyz[0])
        if self.isRender:
            self.camera.move_and_look_at(0,0,0,self.robot.body_xyz[0], self.robot.body_xyz[1], 1)
        return obs, r, done, info

    def super_step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        self._alive = float(
            self.robot.alive_bonus(
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
            self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        ))  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(self._alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}


class MyCamera(Camera):
    m_lookat = [0,0,0]
    m_decay = 0.05
    def move_and_look_at(self, i, j, k, x, y, z):
        lookat = [x, y, z]
        lookat = [self.m_lookat[i]*self.m_decay + (1-self.m_decay)*lookat[i] for i in range(3)]
        camInfo = self.env._p.getDebugVisualizerCamera()

        # distance = camInfo[10]
        # pitch = camInfo[9]
        # yaw = camInfo[8]
        distance = 3
        pitch = -15
        yaw = 20
        self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)


class CameraOnTop(Camera):
    camera_set = False
    def move_and_look_at(self, i, j, k, x, y, z):
        if not self.camera_set:
            distance = 12
            pitch = -45
            yaw = 0
            lookat = [0,0,0]
            self.env._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
            self.camera_set = True