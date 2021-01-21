from pyrobotdesign_env.environments.robot_locomotion import RobotLocomotionEnv

class RoboEnv(RobotLocomotionEnv):
    def __init__(self, task="FlatTerrainTask", grammar_file = "grammar_apr30.dot", rule_sequence="0,7,1,13,1,2,16,12,13,6,4,19,4,17,5,3,2,16,4,5,18,9,8,9,9,8"):
        super().__init__(task, grammar_file, rule_sequence)
        self.camera_angle = 0

    def _render(self):
        super()._render()
        if hasattr(self, "viewer"):
            self.camera_adjust()



    def camera_adjust(self):
        self.camera_simpy_follow_robot()

    def camera_simpy_follow_robot(self, rotate=False):
        self.viewer.camera_params.distance = 4
        # self.viewer.camera_params.pitch = -10
        if rotate:
            self.camera_angle += 0.3
            self.viewer.camera_params.yaw = (self.camera_angle//60)*60
        else:
            self.viewer.camera_params.yaw = 0

        # _current_x = 0
        # _current_y = 0

        # self.viewer.camera_params.position = [_current_x, _current_y, 0.7]

#           viewer.camera_params.position = base_tf[:3,3]
#   viewer.camera_params.yaw = np.pi / 12
#   viewer.camera_params.pitch = -np.pi / 12
#   #viewer.camera_params.yaw = np.pi / 3
#   #viewer.camera_params.pitch = -np.pi / 6
#   viewer.camera_params.distance = 2.0 * np.linalg.norm(upper - lower)