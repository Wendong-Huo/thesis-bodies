import numpy as np
from common.wrapper import WalkerWrapper


class MisalignedWalkerWrapper(WalkerWrapper):
    def observation(self, observation):
        """ Align observations! """
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."

        if observation.shape[0] < 8 + self.max_num_joint*2 + self.max_num_feet:
            observation = np.concatenate((observation, [0]*((self.max_num_joint*2 + self.max_num_feet)-(self.num_joint*2 + self.num_feet))))

        return observation


class RandomAlignedWalkerWrapper(WalkerWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.align_idx_set = {  # a random constant for now
            "walker2d":     [17, 22, 4, 2, 14, 24, 0, 5, 8, 7, 27, 28, 15, 25, 12, 9, 20, 3, 29, 16, 10, 6, 1, 13, 18, 23, 11, 26, 21, 19],
            "ant":          [12, 28, 8, 29, 22, 6, 15, 11, 13, 1, 16, 17, 20, 7, 10, 23, 5, 24, 3, 19, 25, 2, 0, 27, 26, 9, 14, 18, 4, 21],
            "hopper":       [24, 12, 0, 15, 23, 5, 20, 14, 6, 9, 27, 7, 3, 8, 19, 28, 10, 17, 25, 21, 1, 18, 13, 26, 2, 16, 29, 4, 11, 22],
            "halfcheetah":  [18, 12, 7, 24, 2, 25, 14, 23, 10, 17, 20, 0, 16, 22, 8, 5, 27, 9, 6, 21, 26, 11, 15, 13, 1, 28, 29, 3, 4, 19],
        }
        self.align_idx = self.align_idx_set[self.env_name]
        print(f"Random Wrapper: {self.align_idx}")

    def observation(self, observation):
        """ Align observations! """
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."

        if observation.shape[0] < 8 + self.max_num_joint*2 + self.max_num_feet:
            observation = np.concatenate((observation, [0]*((self.max_num_joint*2 + self.max_num_feet)-(self.num_joint*2 + self.num_feet))))

        observation = observation[self.align_idx]

        return observation
