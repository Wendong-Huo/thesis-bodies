import time
import utils
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

if __name__ == "__main__":  # noqa: C901
    bodies = list(range(5))

    eval_env = SubprocVecEnv([utils.make_env(robot_body=f"10{i}") for i in bodies])
    eval_env.reset()
    eval_env.env_method("show_body_id")
    eval_env.env_method("set_view")
    time.sleep(10000)
