from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import common
import common.gym_interface as gym_interface

if __name__ == "__main__":
    hyperparams = common.load_hyperparameters(conf_name="SAC")
    
    venv = DummyVecEnv([gym_interface.make_env(robot_body=300)])

    keys_remove = ["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        if key in hyperparams:
            del hyperparams[key]


    model = SAC('MlpPolicy', venv, verbose=1, seed=common.seed, **hyperparams)
    model.save("output_data/tmp/tmp")

    model = SAC.load("output_data/tmp/tmp.zip")
    model = SAC.load("output_data/models/best_model.zip")
