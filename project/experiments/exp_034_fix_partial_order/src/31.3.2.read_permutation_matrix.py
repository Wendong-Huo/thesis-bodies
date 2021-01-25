import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from common.pns import PNSPPO, permutation_matrix
from common import common, gym_interface, wrapper_custom_align

import matplotlib.pyplot as plt

args = common.args

default_wrapper = [wrapper_custom_align.CustomAlignWrapper]
for rank_idx, test_body in enumerate(args.test_bodies):
    eval_venv = DummyVecEnv([gym_interface.make_env(rank=rank_idx, seed=common.seed, wrappers=default_wrapper, force_render=args.render,
                                                robot_body=test_body,
                                                dataset_folder=args.body_folder)])
    model = PNSPPO.load("output_data/tmp/model-399-499-599-699-CustomAlignWrapper-mdd41d8cd98f00b204e9800998ecf8427e-pns-pns_init-sd80966/best_model.zip", env=eval_venv)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    for i in range(4):
        weight = model.policy.features_extractor.pns[i].weight.detach().cpu().numpy()
        bias = model.policy.features_extractor.pns[i].bias.detach().cpu().numpy()
        a = permutation_matrix(weight)
        print(weight)
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(weight, cmap="gray")
        axes[1].imshow(a, cmap="gray")

        plt.show()

    break