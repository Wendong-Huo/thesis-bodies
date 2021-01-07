import numpy as np
from common import common
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

        self.obs_header_length = self.obs_length - self.max_num_joint*2 - self.max_num_feet

        assert common.args.preserve_header or not common.args.preserve_feet_contact, "preserve_feet_contact should only be used when preserve_header is used."
        if common.args.preserve_header:
            if common.args.preserve_feet_contact:
                self.align_idx = list(range(self.max_num_joint*2))
            else:
                self.align_idx = list(range(self.max_num_joint*2 + self.max_num_feet))
        else:
            self.align_idx = list(range(self.obs_length))


        with temp_seed(common.seed):
            np.random.shuffle(self.align_idx)

        with temp_seed(self.robot.robot_id):
            np.random.shuffle(self.align_idx)
        
        if common.args.random_even_same_body:
            with temp_seed(env.rank):
                np.random.shuffle(self.align_idx)


        print(f"Random Wrapper: {self.align_idx}")

    def observation(self, observation):
        """ Align observations! """
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."

        if observation.shape[0] < 8 + self.max_num_joint*2 + self.max_num_feet:
            observation = np.concatenate((observation, [0]*((self.max_num_joint*2 + self.max_num_feet)-(self.num_joint*2 + self.num_feet))))

        if common.args.preserve_header:
            if common.args.preserve_feet_contact:
                observation = np.concatenate((observation[:self.obs_header_length], observation[self.obs_header_length:self.obs_header_length+self.max_num_joint*2][self.align_idx], observation[self.obs_header_length+self.max_num_joint*2:]))
            else:
                observation = np.concatenate((observation[:self.obs_header_length], observation[self.obs_header_length:][self.align_idx]))
        else:
            observation = observation[self.align_idx]

        return observation
