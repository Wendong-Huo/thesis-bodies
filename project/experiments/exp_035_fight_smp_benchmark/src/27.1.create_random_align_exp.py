import numpy as np
from common import alignments

for seed in range(20):

    str_bodies = ",".join([str(x) for x in np.arange(start=900,stop=908)])
    alignments_to_test = alignments.get_alignments_to_test(num_tests=1, seed=seed, num_envs=16)
    str_cmd = f"python 1.train.py --seed={seed} --train_bodies={str_bodies} --test_bodies={str_bodies} --custom_alignment={alignments_to_test[0]} --topology_wrapper=CustomAlignWrapper --custom_align_max_joints=8"



    print(str_cmd)
    print("")