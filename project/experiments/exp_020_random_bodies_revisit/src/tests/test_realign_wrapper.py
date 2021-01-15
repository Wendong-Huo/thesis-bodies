import unittest
# hack start: for start standalone for debugging.
import os
import sys
if sys.path[0].split("/")[-1]=="tests":
    sys.path.insert(0, os.getcwd())
# hack end
import numpy as np

from common import common
from common.wrapper import ReAlignedWrapper
from common.gym_interface import make_env

class TestReAlignedWrapper(unittest.TestCase):

    def test_realign_idx(self):

        def _get_zipped(method, body_id):
            common.args.realign_method = method
            env = make_env(robot_body=body_id, wrappers=[ReAlignedWrapper])()
            ids = np.arange(len(env.realign_idx))
            zipped = list(zip(ids, env.realign_idx))
            sorted_zipped = list(zip(ids, sorted(env.realign_idx)))
            return zipped, sorted_zipped, env
        
        def _slice(obs, method, feet):
            if method=="aligned":
                return obs
            if method=="general_only":
                return obs[8:]
            if method=="joints_only":
                return np.concatenate((obs[:8],obs[-feet:]))
            if method=="feetcontact_only":
                return obs[:-feet]
            if method=="general_joints":
                return obs[-feet:]
            if method=="general_feetcontact":
                return obs[8:-feet]
            if method=="joints_feetcontact":
                return obs[:8]
            if method=="general_joints_feetcontact":
                return []
            raise NotImplementedError

        test_cases = {300:2, 400:6, 500:4, 600:1}
        methods = ["aligned", "general_only", "joints_only", "feetcontact_only", "general_joints", "general_feetcontact", "joints_feetcontact", "general_joints_feetcontact"]
        for method in methods:
            for body_id, feet in test_cases.items():
                zipped, sorted_zipped, env = _get_zipped(method, body_id)
                assert env.observation_space.shape[0]==len(zipped), "length not match"
                assert env.observation_space.shape[0]==len(sorted_zipped), "length not match"
                
                for a,b in sorted_zipped:
                    assert a==b, f"sorted_zipped: {method}.\n\n {sorted_zipped}"
                zipped = _slice(zipped, method, feet)
                for a,b in zipped:
                    assert a==b, f"zipped: {method}. \n\n {zipped}"

    def test_realigned_obs(self):
        def _get_zipped(method, body_id):
            common.args.realign_method = method
            env = make_env(robot_body=body_id, wrappers=[ReAlignedWrapper])()
            realigned_obs = env.observation(np.arange(env.observation_space.shape[0]))
            return list(zip(realigned_obs, env.realign_idx)), env
            
        test_cases = {300:2, 400:6, 500:4, 600:1}
        methods = ["aligned", "general_only", "joints_only", "feetcontact_only", "general_joints", "general_feetcontact", "joints_feetcontact", "general_joints_feetcontact"]
        for method in methods:
            for body_id, feet in test_cases.items():
                zipped, env = _get_zipped(method, body_id)
                assert env.observation_space.shape[0]==len(zipped), "length not match"
                for a,b in zipped:
                    assert a==b, f"zipped obs: {method}. \n\n {zipped}"

if __name__ == '__main__':
    unittest.main()