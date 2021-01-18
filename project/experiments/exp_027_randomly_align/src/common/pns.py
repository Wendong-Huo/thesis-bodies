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

def debug(grad):
    # print(grad.shape)
    # print(grad[:3,:3])
    # return grad * 1000
    pass
class PNSFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.total_available_modules = 16
        _pns = []
        for i in range(self.total_available_modules):
            alignment_matrix = nn.Linear(observation_space.shape[0]-8, observation_space.shape[0]-8)
            if common.args.pns_init:
                with th.no_grad():
                    alignment_matrix.weight[:,:] = 0.
                    alignment_matrix.bias[:] = 0.
                    for i in range(alignment_matrix.weight.shape[0]):
                        alignment_matrix.weight[i,i] = 1.
                # # For pybullet envs,
                # # first 8 numbers should be global observation (a reasonable prior), so initialize this, might make learning faster.
                # with th.no_grad():
                #     alignment_matrix.weight[:, :8] = 0.
                #     alignment_matrix.weight[:8, :] = 0.
                #     for i in range(8):
                #         alignment_matrix.weight[i,i] = 1.
            _pns.append(alignment_matrix)
        self.pns = nn.ModuleList(_pns)
        self.robot_id_2_idx = {}
        
    def forward(self, observations: th.Tensor, robot_id) -> th.Tensor:
        if isinstance(robot_id, list):
            assert observations.shape[0] == len(robot_id), "Need robot_id for each piece of obs"
            for i in robot_id:
                if i not in self.robot_id_2_idx:
                    if len(self.robot_id_2_idx)==0:
                        next_idx = 0
                    else:
                        next_idx = max(self.robot_id_2_idx.values())+1
                    assert next_idx < self.total_available_modules, "Too many bodies, not enough PNS modules."
                    self.robot_id_2_idx[i] = next_idx

            assert(observations.shape[0]<=16)
            transformed = []
            general_info = observations[:,:8]
            joints_info = observations[:,8:]
            for i, single in enumerate(joints_info):
                single = self.pns[self.robot_id_2_idx[robot_id[i]]](single)
                transformed.append(single)
            joints_info = th.stack(transformed, dim=0)
            observations = th.cat([general_info,joints_info], dim=1)
            # assert observations.shape[0] == 8 and observations.shape[-1] == 26, "Only support 9xx for now."
        else:
            # assert observations.shape[0]!=8, "Not such a coincident, batch size is 8? or something is wrong?"
            # debug: why gradient doesn't pass to pns and pns weights don't get updated.
            # print(self.pns[self.robot_id_2_idx[robot_id]].weight.data.numpy()[:2,:2])
            general_info = observations[:,:8]
            joints_info = observations[:,8:]
            joints_info = self.pns[self.robot_id_2_idx[robot_id]](joints_info)
            observations = th.cat([general_info,joints_info], dim=1)
            if not hasattr(self, "is_hooked"):
                # self.pns[0].weight.register_hook(debug)
                self.is_hooked = True
        return observations

class PNSMotorNet(nn.Module):
    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.total_available_modules = 16
        _pns = []
        for i in range(self.total_available_modules):
            _motor_matrix = nn.Linear(action_space.shape[0], action_space.shape[0])
            if common.args.pns_init:
                with th.no_grad():
                    _motor_matrix.weight[:,:] = 0.
                    _motor_matrix.bias[:] = 0.
                    for i in range(_motor_matrix.weight.shape[0]):
                        _motor_matrix.weight[i,i] = 1.
            _pns.append(_motor_matrix)
        self.pns = nn.ModuleList(_pns)
        self.robot_id_2_idx = {}

    def forward(self, action, robot_id):
        if isinstance(robot_id, list):
            assert action.shape[0] == len(robot_id), "Need robot_id for each piece of obs"
            for i in robot_id:
                if i not in self.robot_id_2_idx:
                    if len(self.robot_id_2_idx)==0:
                        next_idx = 0
                    else:
                        next_idx = max(self.robot_id_2_idx.values())+1
                    assert next_idx < self.total_available_modules, "Too many bodies, not enough PNS modules."
                    self.robot_id_2_idx[i] = next_idx

            assert(action.shape[0]<=16)
            transformed = []
            for i, single in enumerate(action):
                single = self.pns[self.robot_id_2_idx[robot_id[i]]](single)
                transformed.append(single)
            action = th.stack(transformed, dim=0)
            # assert action.shape[0] == 8 and action.shape[-1] == 8, "Only support 9xx for now."
        else:
            action = self.pns[self.robot_id_2_idx[robot_id]](action)
            if not hasattr(self, "is_hooked"):
                # self.pns[0].weight.register_hook(debug)
                self.is_hooked = True
        return action

class PNSRolloutBuffer(RolloutBuffer):
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

# for 9xx, there will be eight bodies.
# add a linear module to each obs piece and each action piece.
class PNSMlpPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = PNSFeaturesExtractor, # !! replace the FeatureExtractor, this is the sensor PNS
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.all_robot_ids = []
        self.current_robot_id = self.all_robot_ids
        self.pns_motor_net = PNSMotorNet(action_space = self.action_space)
        self._build(lr_schedule) # build again, otherwise the new pns_motor_net can't get into the list of optimizer.
        
        # Divide PNS modules into another group, so later can apply a different learning rate, such as 1e-4
        all_parameters = list(self.parameters())
        default_parameters = []
        pns_parameters = []
        for i,p in enumerate(all_parameters):
            if len(p.shape)==2 and (p.shape[0] == p.shape[1] == 8 or p.shape[0] == p.shape[1] == 18):
                pns_parameters.append(p)
            else:
                default_parameters.append(p)

        self.optimizer = self.optimizer_class([
                {'params': default_parameters},
                {'params': pns_parameters}
            ], lr=lr_schedule(1), **self.optimizer_kwargs)


    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """
        Preprocess the observation if needed and extract features.

        :param obs:
        :return:
        """
        assert self.features_extractor is not None, "No features extractor was set"
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        obs = self.features_extractor(preprocessed_obs, self.current_robot_id)
        # self.set_robot_id(self.all_robot_ids) # reset current robot_id after process
        return obs

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        # add motor PNS
        mean_actions = self.pns_motor_net(mean_actions, self.current_robot_id)

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

class PNSPPO(PPO):
    def __init__(self, *args, **argv):
        super().__init__(*args, **argv)

    def _setup_model(self) -> None:
        # ActorCriticPolicy part
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = PNSRolloutBuffer(
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

        # PNSPPO part
        for i in range(self.env.num_envs):
            self.policy.all_robot_ids.append(self.env.envs[i].robot.robot_id)

        if hasattr(self, "pns_senser_robot_id_2_idx"):
            for key, value in self.pns_senser_robot_id_2_idx.items():
                self.policy.features_extractor.robot_id_2_idx[int(key)] = value
        if hasattr(self, "pns_motor_robot_id_2_idx"):
            for key, value in self.pns_motor_robot_id_2_idx.items():
                self.policy.pns_motor_net.robot_id_2_idx[int(key)] = value

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Set learning rate for PNS modules
        assert len(self.policy.optimizer.param_groups)==2, "Divide param_groups first, please."
        self.policy.optimizer.param_groups[1]['lr'] = 1e-4

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
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if eval_env is None:
            eval_env = self.env
        
        assert eval_env.num_envs==1, "More than 1 environment"
        self.policy.set_robot_id(eval_env.envs[0].robot.robot_id)
        return self.policy.predict(observation, state, mask, deterministic)

    def save(self, path, exclude = None, include = None):
        self.pns_senser_robot_id_2_idx = self.policy.features_extractor.robot_id_2_idx
        self.pns_motor_robot_id_2_idx = self.policy.pns_motor_net.robot_id_2_idx
        return super().save(path, exclude=exclude, include=include)

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
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            self.policy.set_robot_id(self.policy.all_robot_ids) # reset robot id before collecting rollouts
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
            if isinstance(model, PNSPPO):
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
