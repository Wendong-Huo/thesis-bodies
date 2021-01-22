import numpy as np
import pybullet
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

class MyWalkerBase(WalkerBase):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        super().__init__(fn, robot_name, action_dim, obs_dim, power)
        # e.g. fn = "".../300.xml"
        self.robot_id = int(fn.split("/")[-1].split(".")[0])

class MyWalkerBaseBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, robot, render=False):
        self._last_x = 0
        self._history_x = []
        self._history_dx = []
        self.colored = False
        super().__init__(robot, render=render)
    
    def step(self, a):
        obs, reward, done, info = super().step(a)
        info["distance_x"] = self.robot.body_xyz[0]
        self.camera_adjust()
        return obs, reward, done, info

    def reset(self):
        self._history_x = []
        self._history_dx = []
        obs = super().reset()
        self.pybullet = self._p
        self.camera_angle = 0

        # self.electricity_cost = 2.0     # hack, encourage exploring! 
                                        # We are wasting computation power and our lives while sitting there waiting them to simulate "saving energy".
                                        # we could encourage energe saving AFTER we have a steady gait.

        if self.isRender and not self.colored and hasattr(self, "pybullet"):
            # only reset color once after the scene is set.
            self._reset_color()
            self.colored = True
        return obs

    def _reset_color(self):
        # to change lighting condition, need an arbitrary flag.
        self.pybullet.configureDebugVisualizer(flag=self.pybullet.COV_ENABLE_MOUSE_PICKING, enable=1,lightPosition=[10,-10,10])

    def show_body_id(self):
        if self._p:
            self._p.addUserDebugText(f"{self.xml.split('/')[-1]}", [-0.5, 0, 1], [0, 0, 1])

    def camera_adjust(self):
        self.camera_simpy_follow_robot()

    def camera_simpy_follow_robot(self, rotate=True):
        if self._p:
            self.camera_angle += 0.3
            distance = 4
            pitch = -10
            if rotate:
                yaw = (self.camera_angle//60)*60
            else:
                yaw = 0

            # Why I need to '*1.1' here?
            _current_x = self.robot.body_xyz[0] * 1.1
            _current_y = self.robot.body_xyz[1] * 1.1

            lookat = [_current_x, _current_y, 0.7]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def camera_follow_robot(self):
        if self._p:
            distance = 4
            pitch = -5
            yaw = 0

            # Smooth Camera
            if len(self._history_x) > 0:
                self._last_x = self._history_x[-1]
            self._history_x.append(self.robot.body_xyz[0])
            self._history_dx.append(self.robot.body_xyz[0] - self._last_x)
            _average_speed = np.mean(self._history_dx) if len(self._history_dx) <= 11 else np.mean(self._history_dx[-10:])
            _current_x = self._last_x + _average_speed
            # print(_current_x, self.robot.body_xyz[0])

            lookat = [_current_x, 0, 0.7]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)

    def set_view(self):
        if self._p:
            distance = 3
            pitch = -80
            yaw = 0
            lookat = [0, 0, 0]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, lookat)


