from torch import nn
from torch import Tensor

import common.utils as utils

class MyThreshold(nn.Threshold):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(utils.args.threshold_threshold, utils.args.threshold_value, inplace)
