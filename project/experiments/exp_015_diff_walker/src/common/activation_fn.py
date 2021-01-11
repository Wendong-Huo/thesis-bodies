from torch import nn

import common.common as common

class MyThreshold(nn.Threshold):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(common.args.threshold_threshold, common.args.threshold_value, inplace)
