import numpy as np
import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

class MyWalkerBase(WalkerBase):
    pass

class MyWalkerBaseBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, robot, render=False):
        self._last_x = 0
        self._history_x = []
        self._history_dx = []
        super().__init__(robot, render=render)
    def reset(self):
        self._history_x = []
        self._history_dx = []
        obs = super().reset()
        self.pybullet = self._p
        return obs

    def show_body_id(self):
        if self._p:
            self._p.addUserDebugText(f"{self.xml.split('/')[-1]}", [-0.5, 0, 1], [0, 0, 1])

    def camera_adjust(self):
        self.camera_follow_robot()

    def camera_follow_robot(self):
        if self._p:
            distance = 3
            pitch = -40
            yaw = 10

            # Smooth Camera
            if len(self._history_x) > 0:
                self._last_x = self._history_x[-1]
            self._history_x.append(self.robot.body_xyz[0])
            self._history_dx.append(self.robot.body_xyz[0] - self._last_x)
            _average_speed = np.mean(self._history_dx) if len(self._history_dx) <= 11 else np.mean(self._history_dx[-10:])
            _current_x = self._last_x + _average_speed
            # print(_current_x, self.robot.body_xyz[0])

            lookat = [_current_x, 0, 0]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def set_view(self):
        if self._p:
            distance = 3
            pitch = -80
            yaw = 0
            lookat = [0, 0, 0]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)
