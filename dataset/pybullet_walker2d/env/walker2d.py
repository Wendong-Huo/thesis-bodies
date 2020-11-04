import numpy as np

import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

from stable_baselines3.common import logger


class Walker2D(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self, xml, powercoeffs=[1.,1.,1.]):
        self.step_num = 0
        self.powercoeffs = powercoeffs
        WalkerBase.__init__(self, xml, "torso", action_dim=6, obs_dim=22, power=0.40)

    def alive_bonus(self, z, pitch):
        # return +1
        # z is the height of torso center.
        # I can scaffold it as well.
        # Scaffold 1: make the robot straight
        # z > self.initial_z * 0.8 / 1.25 and abs(pitch) < 1.0
        # Scaffold 2: help the robot to stand (in first 100 steps) (pybullet_envs's implementation is giving +1 forever, which could lead to standing still, another local optima)
        if self.step_num< 100:
            b = +1
        else:
            b = +0.1
        return b if z > self.initial_z * 0.64 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        torso_length = 4
        torso_width = 5
        torso_max_power_coef = torso_length * torso_width * torso_width
        joint_names = ["thigh", "leg", "foot"]
        joint_powers = [torso_max_power_coef, 100, 30] # where is 80?
        for joint_name, joint_power, coef in zip(joint_names, joint_powers, self.powercoeffs):
            self.jdict[f"{joint_name}_joint"].power_coef = joint_power * coef
            self.jdict[f"{joint_name}_left_joint"].power_coef = joint_power * coef

# volume of parts
# 4 * 5 * 5 = 100
# 4.5 * 5 * 5 = 112.5
# 5 * 4 * 4 = 80
# 2 * 6 * 6 = 72


class Walker2DEnv(WalkerBaseBulletEnv):

    def __init__(self, xml, param, powercoeffs=[1.,1.,1.], render=False, max_episode_steps=1e3):
        self.robot = Walker2D(xml, powercoeffs)
        self.max_episode_steps = max_episode_steps
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

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
                #see Issue 63: https://github.com/openai/roboschool/issues/63
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
