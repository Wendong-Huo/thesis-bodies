from time import sleep
import numpy as np
import gym

from policies.ppo_with_body_info import PPO_with_body_info
from _train_utils.load_dataset import load_dataset
from _train_utils.eval_environment import get_saved_hyperparams, create_test_env
from utils import output, read_yaml
import arguments
args = arguments.get_args_eval()

def eval_model(seed=0):
    conf = read_yaml(args.conf_yml)
    test_bodies = conf["test_bodies"]
    output(f"test_bodies: {test_bodies}", 1)

    env_id, files, params, body_ids = load_dataset(args.dataset)

    hyperparams, stats_path = get_saved_hyperparams(args.stats_path, norm_reward=False, test_mode=True)
    # HACK
    hyperparams["normalize"] = True

    env_kwargs = {
        "xml": files[args.body_id],
        "param": params[args.body_id],
        "max_episode_steps": args.n_timesteps+1,
        "render": args.render,
    }
    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir="tmp/",
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )
    kwargs = dict(seed=seed)
    model = PPO_with_body_info.load(args.model_zip, env=env, **kwargs)
    obs = env.reset()
    state = None
    episode_reward = 0
    ep_len = 0
    body_x_record = []
    for _step in range(args.n_timesteps):
        action, state = model.predict(obs, state=state, deterministic=True)
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        body_x = env.envs[0].robot.body_xyz[0]
        obs, reward, done, infos = env.step(action)
        episode_reward += reward[0]
        ep_len += 1
        if args.render:
            sleep(0.01)
        if done:
            break
    body_x_record.append(body_x)
    obs = env.close()

if __name__ == "__main__":
    args.dataset = "dataset/walker2d_20_10-v0"
    args.conf_yml = "../results/exp_multi_0_bodies.yml"
    args.stats_path = "../results/logs/multi_body/i0_s2100000/walker2d_20_10-v0_1/walker2d_20_10-v0"
    args.model_zip = "../results/logs/multi_body/i0_s2100000/walker2d_20_10-v0_1/best_model.zip"
    args.n_timesteps = 300
    args.render = True

    for i in range(10):
        args.body_id = i
        eval_model()