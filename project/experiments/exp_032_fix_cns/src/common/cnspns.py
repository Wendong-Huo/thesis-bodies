from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Generator

import time

import torch as th
import gym
from torch import nn
from gym import spaces
import numpy as np
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.utils import safe_mean, explained_variance, get_schedule_fn
from stable_baselines3.common import logger
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import base_class
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.callbacks import BaseCallback

from common import common

class PNSSensorAdaptor(nn.Module):
    def __init__(self, sensor_channel):
        super().__init__()
        self._nets = {}
        self.sensor_channel = sensor_channel

    def build_module_dict(self, robot_ids, obs_dim):
        self.obs_dim = obs_dim
        for robot_id in robot_ids:
            self._add_one_net(robot_id)
        self._rebuild_module_dict()
    def add_one_net(self, robot_id):
        self._add_one_net(robot_id)
        self._rebuild_module_dict()
    def _add_one_net(self, robot_id):
        robot_id = str(robot_id)
        if robot_id not in self._nets:
            net = nn.Linear(self.obs_dim, self.sensor_channel)
            net.weight.pns_type = "sensor"
            net.bias.pns_type = "sensor"
            self._nets[robot_id] = net
    def _rebuild_module_dict(self):
        self.nets = nn.ModuleDict(self._nets)

    def forward(self, obs, robot_id):
        assert self.obs_dim == obs.shape[1], f"Max input dimension is {self.obs_dim}"
        if isinstance(robot_id, list):
            robot_ids = robot_id
            assert len(robot_ids) == obs.shape[0], "Need robot_id for each piece of obs"
            transformed = []
            for i, robot_id in enumerate(robot_ids):
                robot_id = str(robot_id)
                _single = self.nets[robot_id]( obs[i] )
                transformed.append(_single)
            obs = th.stack(transformed, dim=0)
        else:
            robot_id = str(robot_id)
            obs = self.nets[robot_id]( obs )
        return obs
class PNSMotorAdaptor(nn.Module):
    def __init__(self, motor_channel):
        super().__init__()
        self._nets = {}
        self.motor_channel = motor_channel
    def build_module_dict(self, robot_ids, action_dim):
        self.action_dim = action_dim
        for robot_id in robot_ids:
            self._add_one_net(robot_id)
        self._rebuild_module_dict()
    def add_one_net(self, robot_id):
        self._add_one_net(robot_id)
        self._rebuild_module_dict()
    def _add_one_net(self, robot_id):
        robot_id = str(robot_id)
        if robot_id not in self._nets:
            net = nn.Linear(self.motor_channel, self.action_dim)
            net.weight.pns_type = "motor"
            net.bias.pns_type = "motor"
            self._nets[robot_id] = net
    def _rebuild_module_dict(self):
        self.nets = nn.ModuleDict(self._nets)
    def forward(self, action, robot_id):
        if isinstance(robot_id, list):
            robot_ids = robot_id
            assert len(robot_ids) == action.shape[0], "Need robot_id for each piece of action"
            transformed = []
            for i, robot_id in enumerate(robot_ids):
                robot_id = str(robot_id)
                _single = self.nets[robot_id]( action[i] )
                transformed.append(_single)
            action = th.stack(transformed, dim=0)
        else:
            robot_id = str(robot_id)
            action = self.nets[robot_id]( action )
        assert self.action_dim == action.shape[1], f"Max output dimension is {self.action_dim}"
        return action

class CNSPNSRolloutBuffer(RolloutBuffer):
    @staticmethod
    def swap(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1)

    def get(self, env_id, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        
        # don't need to flatten, so we don't loss env information
        # # Prepare the data
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap(self.__dict__[tensor]) # no flatten
                # assert self.__dict__[tensor].shape[0] == 8, "Only support 8 bodies for now"
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size], env=env_id)
            start_idx += batch_size
    def _get_samples(self, batch_inds: np.ndarray, env = None) -> RolloutBufferSamples:
        data = (
            self.observations[env,batch_inds],
            self.actions[env,batch_inds],
            self.values[env,batch_inds],
            self.log_probs[env,batch_inds],
            self.advantages[env,batch_inds],
            self.returns[env,batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class CNSPNSPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        if len(args)==3:
            self.lr_schedule = args[2]
        else:
            self.lr_schedule = kwargs['lr_schedule']
        self.sensor_channel = common.args.cnspns_sensor_channel
        self.motor_channel = common.args.cnspns_motor_channel
        self.training_robot_ids = []
        self.current_robot_id = []
        super().__init__(*args, **kwargs)
        self.action_net = nn.Linear(256, self.motor_channel)
        self.pns_sensor_adaptor = PNSSensorAdaptor(self.sensor_channel)
        self.pns_motor_adaptor = PNSMotorAdaptor(self.motor_channel)

    def divide_and_use_different_learning_rates(self):
        # Divide PNS modules into another group, so later can apply a different learning rate, such as 1e-4
        all_parameters = list(self.parameters())
        default_parameters = []
        pns_parameters = []
        for i,p in enumerate(all_parameters):
            if hasattr(p, "pns_type"):
                pns_parameters.append(p)
            else:
                default_parameters.append(p)

        self.optimizer = self.optimizer_class([
                {'params': default_parameters},
                {'params': pns_parameters}
            ], lr=self.lr_schedule(1), **self.optimizer_kwargs)

    def change_learning_rate_for_pns(self):
        # Set learning rate for PNS modules
        enable_different_learning_rates = False
        if enable_different_learning_rates:
            assert len(self.optimizer.param_groups)==2, "Divide param_groups first, please."
            self.optimizer.param_groups[1]['lr'] = 1e-4

    def add_net_to_adaptors(self, robot_id):
        self.pns_sensor_adaptor.add_one_net(robot_id)
        self.pns_motor_adaptor.add_one_net(robot_id)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Copy and inject sensor adaptors
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        # Inject sensor adaptors
        preprocessed_obs = self.pns_sensor_adaptor(preprocessed_obs, self.current_robot_id)
        obs = self.features_extractor(preprocessed_obs)
        return obs

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Copy and inject motor adaptors
        """
        mean_actions = self.action_net(latent_pi)
        # Inject motor adaptors
        mean_actions = self.pns_motor_adaptor(mean_actions, self.current_robot_id)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")
    
    
    def set_robot_id(self, robot_id):
        self.current_robot_id = robot_id

    def _build_mlp_extractor(self) -> None:
        """
        Change the size of input of mlp_extractor to [sensor_channel].
        """
        self.mlp_extractor = MlpExtractor(
            self.sensor_channel, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

class CNSPNSPPO(PPO):
    need_env_id = True
    def _setup_model(self) -> None:
        # ActorCriticPolicy part
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = CNSPNSRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        
        # PPO part
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # CNSPNSPPO part
        # plug in adaptors for each body
        robot_ids = set()
        training_robot_ids = []
        for i in range(self.env.num_envs):
            robot_ids.add(self.env.envs[i].robot.robot_id)
            training_robot_ids.append(self.env.envs[i].robot.robot_id)
        self.policy.training_robot_ids = training_robot_ids
        self.policy.pns_sensor_adaptor.build_module_dict(robot_ids, self.observation_space.shape[0])
        self.policy.pns_motor_adaptor.build_module_dict(robot_ids, self.action_space.shape[0])
        self.policy.divide_and_use_different_learning_rates()

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Copy and inject
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # Inject
            self.policy.set_robot_id(self.policy.training_robot_ids) # reset robot id before collecting rollouts

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
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
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Copy and inject
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        self.policy.change_learning_rate_for_pns()

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for env_id in range(self.n_envs):
                # Inject
                self.policy.set_robot_id(self.env.envs[env_id].robot.robot_id)
                # Do a complete pass on the rollout buffer
                for rollout_data in self.rollout_buffer.get(env_id=env_id, batch_size=self.batch_size):
                    
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # TODO: investigate why there is no issue with the gradient
                    # if that line is commented (as in SAC)
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

                all_kl_divs.append(np.mean(approx_kl_divs))

                if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                    break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
        eval_env = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Copy and inject
        """
        if eval_env is None:
            eval_env = self.env
        
        assert eval_env.num_envs==1, "More than 1 environment"
        # Inject
        self.policy.set_robot_id(eval_env.envs[0].robot.robot_id)
        return self.policy.predict(observation, state, mask, deterministic)
    
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy"] # no need to save "policy.optimizer"
        return state_dicts, []



# standalone function

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
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
            if hasattr(model, "need_env_id") and model.need_env_id:
                action, state = model.predict(obs, state=state, deterministic=deterministic, eval_env=env)
            else: # for default use, EvalCallback_with_prefix calls this by default
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
