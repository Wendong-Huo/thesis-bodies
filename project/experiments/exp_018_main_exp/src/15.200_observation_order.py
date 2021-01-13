# Maximum joint number is 10.

# a0[0,1,2,3,4,5,6,7,8,9]
# a1[0,1,2,3,4,5,6,7,8,9]
# ...
# a15[0,1,2,3,4,5,6,7,8,9]

# Total possible combinations = (Pr10)^16 = (10!)^16 = 9e104 (A huge number.)

# So we randomly shuffle them, and if repeats, re-shuffle to get a new one.

import numpy as np
from common import seeds

with seeds.temp_seed(0):
    for j in range(2):
        joint_orders = []
        for i in range(16):
            joint_order = np.arange(10)
            np.random.shuffle(joint_order)
            joint_orders.append(joint_order)
        joint_orders = np.array(joint_orders)
        print(joint_orders)