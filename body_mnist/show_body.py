import time
import utils
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

if __name__ == "__main__":  # noqa: C901
    bodies = [int(x) for x in utils.args.train_bodies.split(',')]
    print(bodies)
    envs = [utils.make_env(robot_body=i, body_info=0) for i in bodies]
    eval_env = SubprocVecEnv(envs)
    eval_env.reset()
    eval_env.env_method("show_body_id")
    eval_env.env_method("set_view")
    time.sleep(10000)
