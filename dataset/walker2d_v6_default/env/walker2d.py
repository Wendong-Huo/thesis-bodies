import numpy as np

import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase, Walker2D

from stable_baselines3.common import logger

from pybullet_envs.env_bases import Camera

class Walker2D_deleted(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self, xml, param, env, powercoeffs=[1., 1., 1.]):
        self.step_num = 0
        self.env = env
        self.xml = xml
        self.param = param
        self.powercoeffs = powercoeffs
        WalkerBase.__init__(self, xml, "torso", action_dim=6, obs_dim=22, power=0.40)

    def alive_bonus(self, z, pitch):
        # z is the height of torso center.

        # I can scaffold it as well.
        # Scaffold 1: make the robot straight
        straight = z > self.initial_z * 0.64 and abs(pitch) < 1.0
        # z > self.initial_z * 0.8 / 1.25 and abs(pitch) < 1.0
        # Scaffold 2: help the robot to stand (in first 100 steps) (pybullet_envs's implementation is giving +1 forever, which could lead to standing still, another local optima)
        b = +1 if self.step_num < 100 else +0.1
        
        return b if straight else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        # Adjust power coefficients so that motors won't exert power that is too large or too small.
        # Power that is too large will cause PyBullet to have the "fly-away" bug.
        # Power that is too small won't be possible to achieve the locomotion task.
        empirical_c1 = 2e4
        empirical_c2 = 2e4 # the torso is especially prone to the "fly-away" bug, so this number should keep small.
        # power coefficients is proportional to the volume of adjacent parts.
        thigh_joint_power_coef = (self.param["volume_torso"] + self.param["volume_thigh"]) * empirical_c1
        leg_joint_power_coef = (self.param["volume_thigh"] + self.param["volume_leg"]) * empirical_c1
        foot_joint_power_coef = (self.param["volume_leg"] + self.param["volume_foot"]) * empirical_c1
        # avoid too large
        torso_max_power_coef = self.param["volume_torso"] * empirical_c2
        thigh_joint_power_coef = min(torso_max_power_coef, thigh_joint_power_coef)
        # set them
        joint_names = ["thigh", "leg", "foot"]
        joint_powers = [thigh_joint_power_coef, leg_joint_power_coef, foot_joint_power_coef]
        for joint_name, joint_power, coef in zip(joint_names, joint_powers, self.powercoeffs):
            self.jdict[f"{joint_name}_joint"].power_coef = joint_power * coef
            self.jdict[f"{joint_name}_left_joint"].power_coef = joint_power * coef


class Walker2DEnv(WalkerBaseBulletEnv):

    def __init__(self, xml, param, powercoeffs=[1., 1., 1.], is_eval=False, render=False, max_episode_steps=1e3):
        self.robot = Walker2D()
        self.is_eval = is_eval
        self.max_episode_steps = max_episode_steps
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
        self.camera = MyCamera(self)

    def reset(self):
        logger.record("debug/body_x", self.robot.body_xyz[0])
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
        if done and self.is_eval:
            logger.record(f"eval/body_x", self.robot.body_xyz[0])
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
