from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common.callbacks import EventCallback, EvalCallback
from policies.policy_with_bodyinfo import PolicyWithBodyInfo
from policies.ppo_without_body_info import PPO_without_body_info
from policies.ppo_with_body_info import PPO_with_body_info

from utils import output
from torch.nn import Linear
import matplotlib.pyplot as plt
from time import time
import os
import shutil
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

def evaluate_policy_with_bodyinfo(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """ Copied and modified from: stable_baselines3.common.evaluation.evaluate_policy"""

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            """add bodyinfo"""
            if isinstance(model, PPO_with_body_info):
                action, state = model.predict_with_bodyinfo(obs, env=env, state=state, deterministic=deterministic)
            else:
                action, state = model.predict(obs, state=state, deterministic=deterministic)

            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

class EvalCallback_with_bodyinfo(EvalCallback):
    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy_with_bodyinfo(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

class DumpWeightsCallback(EventCallback):
    def _on_training_start(self):
        self.folder = "outputs/DumpWeightsCallback"
        if isinstance(self.model, PPO_with_body_info):
            self.folder += "_body"
        shutil.rmtree(self.folder, ignore_errors=True)
        os.makedirs(self.folder, exist_ok=True)
        
    def _on_rollout_start(self) -> None:
        output(f"Dumping weights of {self.model}", 2)
        if isinstance(self.model.policy, PolicyWithBodyInfo):
            layer = self.model.policy.bodyinfo_linear1
            if isinstance(layer, Linear):
                plt.figure(figsize=[10, 10])
                ndarray = layer.weight.detach().numpy()
                # output(f"Sum of layer1 {np.sum(ndarray, keepdims=False)}",2)
                # print(ndarray[0,:5])
                plt.imshow(ndarray, cmap="gray")
                plt.colorbar()
                img_path = f"{self.folder}/Layer_b1_{self.model.num_timesteps}.png"
                plt.savefig(img_path)
                # output(f"Writing {img_path}", 2)
                plt.close()
            pass

        policy_net = self.model.policy.mlp_extractor.policy_net._modules
        for key in policy_net:
            layer = policy_net[key]
            if isinstance(layer, Linear):
                plt.figure(figsize=[10, 10])
                plt.imshow(layer.weight.detach().numpy(), cmap="gray")
                plt.colorbar()
                img_path = f"{self.folder}/Layer_{key}_{self.model.num_timesteps}.png"
                plt.savefig(img_path)
                output(f"Writing {img_path}")
                plt.close()
