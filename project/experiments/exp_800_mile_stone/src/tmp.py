import gym
import numpy as np
from common import gym_interface

bodies = np.arange(300,399)
print(gym_interface.get_max_num_joints(bodies))
