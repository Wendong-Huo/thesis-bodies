import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from torch.nn.modules.activation import ReLU

from utils import output

class PolicyWithBodyInfo(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        device: Union[th.device, str] = "auto",
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(PolicyWithBodyInfo, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
        self.bodyinfo_linear1 = nn.Linear(7, 16) # for Ant
        self.bodyinfo_activation = nn.ReLU()
        self.bodyinfo_linear2 = nn.Linear(16, 7) # divide into 7 categories, so I don't need to change the mlp input size
        self.bodyinfo_softmax = nn.Softmax(dim=1)
        self._build(lr_schedule) # need to call this to add these weights into update schedule
        output("MlpPolicy Initialized.", 2)

    def bodynet(self, obs):
        """ Turn body params into the probability of being in one of seven categories """
        (raw_obs,bodyinfo) = th.split(obs, [28,7], dim=1) # this number is for Ant
        x = self.bodyinfo_linear1(bodyinfo)
        x = self.bodyinfo_activation(x)
        x = self.bodyinfo_linear2(x)
        x = self.bodyinfo_softmax(x)
        obs = th.cat([raw_obs, x], dim=1)
        return obs

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        obs = self.bodynet(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        We need modify this function as well in order to get these additional weights trained.
        """
        obs = self.bodynet(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
