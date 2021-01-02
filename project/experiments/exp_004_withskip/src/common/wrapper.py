import gym
from gym.spaces.space import Space
import numpy as np

from gym_envs.ant import MyAntBulletEnv
from gym_envs.walker2d import MyWalker2DBulletEnv

class BodyinfoWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, bodyinfo=0):
        # one-hot encoding length for bodyinfo: 10
        self.bodyinfo_length = 2

        gym.ObservationWrapper.__init__(self, env)
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        obs_length = len(self.old_obs_space.high) + self.bodyinfo_length
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[obs_length])

        self.bodyinfo = np.zeros(shape=[self.bodyinfo_length], dtype=np.float32)
        self.bodyinfo[bodyinfo] = 1

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.concatenate((obs, self.bodyinfo))
        return obs

class WalkerWrapper(gym.ObservationWrapper):
    """ unify ant and walker2d, align the observation and action """
    def __init__(self, env: gym.Env):
        if "HopperBulletEnv" in env.spec._env_name:
            self.env_name = "hopper"
            self.num_joint = 3
        elif "HalfCheetahBulletEnv" in env.spec._env_name:
            self.env_name = "halfcheetah"
            self.num_joint = 6
        elif "AntBulletEnv" in env.spec._env_name:
            self.env_name = "ant"
            self.num_joint = 8
        elif "Walker2DBulletEnv" in env.spec._env_name:
            self.env_name = "walker2d"
            self.num_joint = 6
        else:
            raise NotImplementedError
        self.num_feet = len(env.robot.foot_list)

        self.max_num_joint = 8
        self.max_num_feet = 6

        gym.ObservationWrapper.__init__(self, env)
        # re-calculate observation space size
        self.old_obs_space = self.observation_space
        low = self.old_obs_space.low[0]
        high = self.old_obs_space.high[0]
        obs_length = 8 + self.max_num_joint*2 + self.max_num_feet
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=[obs_length])
        # re-calculate action space size
        self.old_action_space = self.action_space
        low = self.old_action_space.low[0]
        high = self.old_action_space.high[0]
        self.action_space = gym.spaces.Box(low=low, high=high, shape=[self.max_num_joint])


    def step(self, action):
        if self.num_joint < self.max_num_joint:
            action = action[:self.num_joint]
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        assert observation.shape[0] == 8 + self.num_joint*2 + self.num_feet, "WalkerWrapper: observation dimension error."
        if self.num_joint < self.max_num_joint:
            observation = np.concatenate( (observation[:8+self.num_joint*2] , [0]*(2*(self.max_num_joint-self.num_joint)), observation[8+self.num_joint*2:] ))
        if self.num_feet < self.max_num_feet:
            observation = np.concatenate( (observation, [0]*(self.max_num_feet-self.num_feet) ))
        return observation


def getSkipFrameWrapper(skip_frames=3):
    class SkipFrameWrapper(gym.Wrapper):
        num_frames_skip = skip_frames
        def __init__(self, env):
            super().__init__(env)
        
        def step(self, action):
            # skip several frames with the same action
            for i in range(self.num_frames_skip):
                super().step(action)
            return super().step(action)
    return SkipFrameWrapper