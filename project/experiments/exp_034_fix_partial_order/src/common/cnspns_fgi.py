from torch import nn
import torch as th
from common import cnspns

class PNSSensorAdaptor_fix_general_info(cnspns.PNSSensorAdaptor):
    # The first 8 numbers in obs are the general information, and the order of them always fixed.
    num_general_info = 8

    def __init__(self, sensor_channel):
        super().__init__(sensor_channel)
        assert sensor_channel>8, "Not enough channel even for the general information"

    def _add_one_net(self, robot_id):
        robot_id = str(robot_id)
        if robot_id not in self._nets:
            net = nn.Linear(self.obs_dim - self.num_general_info, self.sensor_channel - self.num_general_info)
            net.weight.pns_type = "sensor"
            net.bias.pns_type = "sensor"
            net.weight.robot_id = robot_id
            net.bias.robot_id = robot_id
            self._nets[robot_id] = net

    def forward(self, obs, robot_id):
        assert self.obs_dim == obs.shape[1], f"Max input dimension is {self.obs_dim}"
        obs_fgi = obs[:,:self.num_general_info]
        obs_bodily = obs[:, self.num_general_info:]

        obs_bodily = self._transform_obs(obs_bodily, robot_id)

        obs = th.cat([obs_fgi, obs_bodily], dim=1)
        return obs

