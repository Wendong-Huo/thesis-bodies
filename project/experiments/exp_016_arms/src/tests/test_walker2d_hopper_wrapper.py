import unittest
# hack start: for start standalone for debugging.
import os
import sys
if sys.path[0].split("/")[-1]=="tests":
    sys.path.insert(0, os.path.abspath(__file__ + "/../../"))
    os.chdir(os.path.abspath(__file__ + "/../../"))
# hack end
import numpy as np

from common import common
from common import wrapper_diff
from common.gym_interface import make_env

class TestWalker2dHopperWrapper(unittest.TestCase):
    cases = [None]*7
    cases[1] = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 14, 0, 0, 0, 0, 0, 0, 0]
    cases[2] = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 0, 0, 0, 0, 0, 0, 0]
    cases[3] = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 8, 9, 0, 0, 0, 0, 0, 0, 14, 0]
    cases[4] = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 14, 0]
    cases[5] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 0, 0, 0, 0, 14, 0]
    cases[6] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 0]
    def test_aligned_idx(self):
        for i in np.arange(start=1,stop=7):
            wrapper = getattr(wrapper_diff, f"Walker2DHopperCase{i}")
            env = make_env(robot_body=600, wrappers=[wrapper])()
            assert env.observation_space.shape[0]==22, "Should extend Hopper into Walker2D's Observation space"
            fake_obs = np.arange(start=0,stop=15)
            assert env.observation(fake_obs).tolist()==self.cases[i]



if __name__ == '__main__':
    unittest.main()