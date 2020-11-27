from stable_baselines3.common.callbacks import EventCallback, EvalCallback

from utils import output
from torch.nn import Linear
import matplotlib.pyplot as plt
from time import time
import os
import shutil

class EvalCallback_with_hook(EvalCallback):
    count_times = 0
    def _on_step(self):
        ret = super()._on_step()
        if "eval/mean_ep_length" in self.logger.Logger.CURRENT.name_to_value:
            if self.logger.Logger.CURRENT.name_to_value["eval/mean_ep_length"]>=1000:
                self.count_times += 1
        if self.count_times>10:
            self.model.switch_to_larger_buffer = True
            output("Switch to larger buffer.",2)
        return ret

class DumpWeightsCallback(EventCallback):
    def _on_training_start(self):
        self.folder = "outputs/DumpWeightsCallback"
        shutil.rmtree(self.folder, ignore_errors=True)
        os.makedirs(self.folder, exist_ok=True)
        
    def _on_rollout_start(self) -> None:
        output(f"Dumping weights of {self.model}", 2)
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
