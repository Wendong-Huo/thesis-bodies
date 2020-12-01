import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pathlib
import io

import gym
from gym import spaces
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common import logger
from stable_baselines3.common.utils import explained_variance
from .ppo_without_body_info import PPO_without_body_info
from utils import output
from arguments import get_args_train
from .policy_with_bodyinfo import PolicyWithBodyInfo

args = get_args_train()


def expand_space(space, env, num):
    if env is None:
        observation = space.sample()
    else:
        observation = env.observation_space.sample()
    low = np.full(observation.shape[0] + num, space.low[0], dtype=np.float32)
    high = np.full(observation.shape[0] + num, space.high[0], dtype=np.float32)
    expaned_space = gym.spaces.Box(low, high, dtype=observation.dtype)
    return expaned_space


class PPO_with_body_info(PPO_without_body_info):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(PolicyWithBodyInfo,
                         env,
                         learning_rate,
                         n_steps,
                         batch_size,
                         n_epochs,
                         gamma,
                         gae_lambda,
                         clip_range,
                         clip_range_vf,
                         ent_coef,
                         vf_coef,
                         max_grad_norm,
                         use_sde,
                         sde_sample_freq,
                         target_kl,
                         tensorboard_log,
                         create_eval_env,
                         policy_kwargs,
                         verbose,
                         seed,
                         device,
                         _init_setup_model)

    def _setup_model(self) -> None:
        """with body info"""
        self.n_param = None
        robot_params = []
        # read params from the robot once
        for i in range(self.env.num_envs):
            robot_param = self.env.envs[i].robot.param
            robot_params.append(robot_param)

        self.param = np.zeros(shape=[self.env.num_envs, len(robot_params[0])], dtype=np.float32)
        for i in range(self.env.num_envs):
            j = 0
            for key in robot_params[i]:
                self.param[i, j] = robot_params[i][key]
                j += 1
        # expand obs space
        self.observation_space = expand_space(self.observation_space, self.env, len(robot_params[0]))

        super()._setup_model()

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        if self.n_param is None:
            # self.n_param = np.broadcast_to(self.param, shape=[env.num_envs, self.param.shape[1]])
            self.n_param = self.param
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            """with body info"""
            self._last_expanded_obs = np.concatenate([self._last_obs, self.n_param], axis=1)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_expanded_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_expanded_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            new_expanded_obs = np.concatenate([new_obs, self.n_param], axis=1)
            obs_tensor = th.as_tensor(new_expanded_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def predict_with_bodyinfo(self, observation, env, state, deterministic):
        assert env.num_envs == 1
        expanded_observation = np.concatenate([observation, np.array(list(env.envs[0].robot.param.values())).reshape(1, -1)], axis=1)
        return self.policy.predict(expanded_observation, state, None, deterministic)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        """with body info"""
        if False:
            print("baseline")
            a = self.param
            a = a.flatten()
            print("param:")
            print(f"{a[0]:.03f} {a[1]:.03f} {a[2]:.03f} {a[3]:.03f} {a[4]:.03f} {a[5]:.03f} {a[6]:.03f}")
            (a, _) = self.policy.predict(expanded_observation, state, mask, deterministic)
            a = a.flatten()
            print(f"{a[0]:.03f} {a[1]:.03f} {a[2]:.03f} {a[3]:.03f} {a[4]:.03f} {a[5]:.03f} {a[6]:.03f} {a[7]:.03f}")
            print("probe zeros")
            tmp_expanded_observation = np.concatenate([observation, np.zeros_like(self.param)], axis=1)
            (a, _) = self.policy.predict(tmp_expanded_observation, state, mask, deterministic)
            a = a.flatten()
            print(f"{a[0]:.03f} {a[1]:.03f} {a[2]:.03f} {a[3]:.03f} {a[4]:.03f} {a[5]:.03f} {a[6]:.03f} {a[7]:.03f}")
            print("probe ones")
            tmp_expanded_observation = np.concatenate([observation, np.ones_like(self.param)], axis=1)
            (a, _) = self.policy.predict(tmp_expanded_observation, state, mask, deterministic)
            a = a.flatten()
            print(f"{a[0]:.03f} {a[1]:.03f} {a[2]:.03f} {a[3]:.03f} {a[4]:.03f} {a[5]:.03f} {a[6]:.03f} {a[7]:.03f}")
            print("")
        else:
            # expanded_observation = np.concatenate([observation, np.zeros_like(self.param)], axis=1)
            expanded_observation = np.concatenate([observation, self.param], axis=1)
        return self.policy.predict(expanded_observation, state, mask, deterministic)

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file (modify to avoid confliction)

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params, pytorch_variables = load_from_zip_file(path, device=device)

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            # check_for_correct_spaces(env, data["observation_space"], data["action_space"])
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                recursive_setattr(model, name, pytorch_variables[name])

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model
