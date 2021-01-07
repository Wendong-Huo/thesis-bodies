import numpy as np
from common.wrapper import WalkerWrapper
from common.seeds import temp_seed

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
        self.align_idx = list(range(self.obs_length))
        with temp_seed(self.robot.robot_id):
            np.random.shuffle(self.align_idx)
        print(f"Random Wrapper: {self.align_idx}")

    def observation(self, observation):
        """ Align observations! """
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."

        if observation.shape[0] < 8 + self.max_num_joint*2 + self.max_num_feet:
            observation = np.concatenate((observation, [0]*((self.max_num_joint*2 + self.max_num_feet)-(self.num_joint*2 + self.num_feet))))

        observation = observation[self.align_idx]

        return observation
