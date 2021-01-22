import os
from typing import Optional
from torch import Tensor
import imageio
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
# from stable_baselines3.common.evaluation import evaluate_policy
from common.pns import evaluate_policy
from common import common

class EvalCallback_with_prefix(EvalCallback):
    """Slightly modified version"""
    def __init__(
        self,
        eval_env,
        prefix = "",
        callback_on_new_best=None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        super().__init__(eval_env,
                         callback_on_new_best,
                         n_eval_episodes,
                         eval_freq,
                         log_path,
                         best_model_save_path,
                         deterministic,
                         render,
                         verbose)
        self.prefix = prefix

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(
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
            """ Add prefix """
            # Add to current Logger
            self.logger.record(f"eval/{self.prefix}_mean_reward", float(mean_reward))
            # self.logger.record(f"eval/{self.prefix}_mean_ep_length", mean_ep_length)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                self.update_best_reward(self.eval_env.envs[0].robot.robot_id, self.best_mean_reward)
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True

    # This sets the score for SkipSolvedCallback
    def update_best_reward(self, robot_id, best_mean_reward):
        print(f"UpdateBestReward {best_mean_reward}")
        if not hasattr(self.model, "best_rewards"):
            self.model.best_rewards = {}
        self.model.best_rewards[robot_id] = best_mean_reward
        self.model.robot_id_by_score = [idx for idx,v in sorted(self.model.best_rewards.items(), key=lambda item: item[1])][::-1]
        self.model.skip_robot_ids = []
        if hasattr(self.model, "skip_solved_threshold"):
            if best_mean_reward > self.model.skip_solved_threshold:
                if robot_id not in self.model.skip_robot_ids:
                    self.model.skip_robot_ids.append(robot_id)
            if len(self.model.skip_robot_ids)==len(self.model.best_rewards): # if all bodies pass threshold,
                self.model.skip_robot_ids = []                               # then no body need to be skipped.
        self.model.worst_robot_id = self.model.robot_id_by_score[-1]

class SaveVecNormalizeCallback(BaseCallback):
    """From zoo"""
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: Optional[str] = None, verbose: int = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


# update_best_reward() set the scores, SkipSolvedCallback determine the behavior
class SkipSolvedCallback(BaseCallback):
    def __init__(self, skip_threshold: float=1e3, verbose: int=0):
        self.skip_threshold = skip_threshold
        self.robot_2_env_id = None
        self.next_env_idx = 0
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        if not hasattr(self.model, "skip_solved_threshold"):
            self.model.skip_solved_threshold = self.skip_threshold
        if hasattr(self.model, "robot_id_by_score"):
            if self.robot_2_env_id is None:
                self.robot_2_env_id = {}
                for env_idx, env in enumerate(self.model.env.envs):
                    if env.robot.robot_id in self.robot_2_env_id:
                        self.robot_2_env_id[env.robot.robot_id].append(env_idx)
                    else:
                        self.robot_2_env_id[env.robot.robot_id] = [env_idx]

            if len(self.model.skip_robot_ids)>0:
                worst_ids = self.robot_2_env_id[self.model.worst_robot_id]
                for skip_robot_id in self.model.skip_robot_ids:
                    skip_ids = self.robot_2_env_id[skip_robot_id]
                    for idx in skip_ids:
                        worst_idx = worst_ids[self.next_env_idx%len(worst_ids)]
                        self.next_env_idx += 1
                        # Copy data
                        self.model._last_obs[idx] = self.model._last_obs[worst_idx]
                        self.locals["actions"][idx] = self.locals["actions"][worst_idx]
                        self.locals["rewards"][idx] = self.locals["rewards"][worst_idx]
                        self.model._last_dones[idx] = self.model._last_dones[worst_idx]
                        self.locals["values"][idx] = self.locals["values"][worst_idx]
                        self.locals["log_probs"][idx] = self.locals["log_probs"][worst_idx]
            print(self.robot_2_env_id[self.model.worst_robot_id])
        return True

class InspectionCallback(BaseCallback):
    # I'd like to write the model's graph to tensorboard
    # to be implemented. still learning.

    # let me start by write some PNG files on the disk

    def __init__(self, verbose: int=0):
        super().__init__(verbose=verbose)
        self.sub_dir = ""
        self.saved_sensor_weights = np.zeros([1])
        self.saved_motor_weights = np.zeros([1])
        self.saved_actions = []
        self.distance_x = None
 
    def log_weights_to_disk(self):
        if not common.args.pns: # only needed while pns is enabled
            return
        if self.model.num_timesteps%1000==0:
            if self.sub_dir == "":
                if len(self.logger.Logger.CURRENT.output_formats)==2:
                    tb_dir = self.logger.Logger.CURRENT.output_formats[1].writer.log_dir
                    sub_dir = "/".join(tb_dir.split("/")[-2:])
                    for i in range(8):
                        os.makedirs(f"output_data/saved_images/{sub_dir}/sensor_{i}_weight", exist_ok=True)
                        os.makedirs(f"output_data/saved_images/{sub_dir}/motor_{i}_weight", exist_ok=True)
                    self.sub_dir = f"output_data/saved_images/{sub_dir}"
            
            if self.sub_dir != "":
                if hasattr(self.model.policy.features_extractor, "pns"):
                    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
                    current_sensor_weights = self.model.policy.features_extractor.pns[0].weight.cpu().detach().numpy()
                    current_motor_weights = self.model.policy.pns_motor_net.pns[0].weight.cpu().detach().numpy()
                    if current_sensor_weights.sum() == self.saved_sensor_weights.sum() and current_motor_weights.sum() == self.saved_motor_weights.sum() : # nothing has changed
                        return
                    self.saved_sensor_weights = current_sensor_weights.copy()
                    self.saved_motor_weights = current_motor_weights.copy()

                    for i in range(8):
                        min_value = -1.0; max_value = 1.0
                        sensor_weight = self.model.policy.features_extractor.pns[i].weight.cpu().detach().numpy()
                        if i==0 or i==1:
                            print(f"A tiny bit of sensor_weight\n{sensor_weight[7:10,7:10]}")
                        sensor_weight = (sensor_weight - min_value) / (max_value - min_value) * 255
                        sensor_weight = np.clip(sensor_weight, 0, 255)
                        sensor_weight = sensor_weight.astype(np.uint8)
                        imageio.imsave(f"{self.sub_dir}/sensor_{i}_weight/{self.model.num_timesteps}.png", sensor_weight)

                        motor_weight = self.model.policy.pns_motor_net.pns[i].weight.cpu().detach().numpy()
                        if i==0 or i==1:
                            print(f"A tiny bit of motor_weight\n{motor_weight[:3,:3]}")
                        motor_weight = (motor_weight - min_value) / (max_value - min_value) * 255
                        motor_weight = np.clip(motor_weight, 0, 255)
                        motor_weight = motor_weight.astype(np.uint8)
                        imageio.imsave(f"{self.sub_dir}/motor_{i}_weight/{self.model.num_timesteps}.png", motor_weight)

    def output_action(self):
        self.saved_actions.append(np.abs(self.locals["clipped_actions"]).mean())
        if self.model.num_timesteps%5000==0:
            print(f"Avg action value: {np.mean(self.saved_actions)}")
            self.saved_actions = []

    def log_record_rollout_reward(self):
        for i in range(self.model.n_envs):
            if self.locals['dones'][i]:
                if self.distance_x is None: # lazy init
                    self.distance_x = {}
                    for j in range(self.model.n_envs):
                        self.distance_x[j] = deque()

                self.distance_x[i].append(self.locals['infos'][i]['distance_x'])
                if len(self.distance_x[i])>10:
                    self.distance_x[i].popleft()
                self.logger.record(f'rollout_episodic_distance_x/robot_{self.model.env.envs[i].robot.robot_id}', np.mean(self.distance_x[i]))
                # print(self.distance_x[i])

    def _on_step(self):
        # obs = Tensor(self.locals["new_obs"])
        # self.logger.Logger.CURRENT.output_formats[1].writer.add_graph(self.model.policy, obs, verbose=1)
        # print("Write Torch Graph")
        
        # Creating too many files on the server :D, only enable this when needed.
        self.log_weights_to_disk()
        self.log_record_rollout_reward()

        self.output_action()
        pass