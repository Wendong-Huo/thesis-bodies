import time
import utils
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

if __name__ == "__main__":  # noqa: C901
    bodies = list(range(3))

    eval_env = SubprocVecEnv([utils.make_env(robot_body=i) for i in bodies])
    eval_env.reset()
    eval_env.env_method("show_body_id")
    time.sleep(10000)
