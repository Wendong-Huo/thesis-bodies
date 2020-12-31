from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface
if __name__ == "__main__":

    print(str(common.output_data_folder / "tensorboard"))
    print(str(common.input_data_folder / "bodies"))

    args = common.args

    # PPO.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(common.seed)

    hyperparams = common.load_hyperparameters()
    print(hyperparams)

    # Make every env has the same obs space and action space
    default_wrapper = wrapper.WalkerWrapper

    assert len(args.train_bodies)>0, "No body to train."
    env = DummyVecEnv([gym_interface.make_env(rank=i, seed=common.seed, wrapper=default_wrapper, render=args.render,
                                              robot_body=args.train_bodies[i % len(args.train_bodies)], body_info=0) for i in range(args.num_venvs)])

    obs = env.reset()
    print(obs)