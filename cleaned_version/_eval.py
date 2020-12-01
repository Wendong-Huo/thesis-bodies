from time import sleep
import numpy as np
import gym

from policies.ppo_with_body_info import PPO_with_body_info
from policies.ppo_without_body_info import PPO_without_body_info
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
    if args.body:
        model = PPO_with_body_info.load(args.model_zip, env=env, **kwargs)
    else:
        model = PPO_without_body_info.load(args.model_zip, env=env, **kwargs)
    obs = env.reset()
    env.envs[0].env._p.addUserDebugText("x", textPosition=[10,0,0], textColorRGB=[1,0,0])
    env.envs[0].env._p.addUserDebugText("y", textPosition=[0,10,0], textColorRGB=[1,0,0])
    state = None
    episode_reward = 0
    ep_len = 0
    body_x_record = []
    for _step in range(args.n_timesteps):
        action, state = model.predict(obs, state=state, deterministic=True)
        body_x = env.envs[0].robot.body_xyz[0]
        obs, reward, done, infos = env.step(action)
        episode_reward += reward[0]
        ep_len += 1
        if _step%100==0:
            output(f"{ep_len}) body_x: {body_x:.04f}. reward: {reward[0]:.04f}, {episode_reward:.04f}", 2)
        if args.render:
            sleep(0.002)
            pass
        if done:
            break
    body_x_record.append(body_x)
    obs = env.close()
    return body_x_record

if __name__ == "__main__":
    dataset_version = "ant_20_100-v0"
    args.dataset = f"dataset/{dataset_version}"
    args.body = False
    if args.body:
        nn = "_body"
        output(f"evaluating with_body",1)
    else:
        nn = ""
        output(f"evaluating without_body",1)

    # path = f"outputs/{dataset_version}"
    # path1 = f"{path}/logs/multi_body/i0_s3000000/{dataset_version}_1"
    args.conf_yml = f"outputs/ant_20_100-v0/exp_multi_0_bodies.yml"

    args.stats_path = f"outputs/ant_20_100-v0/logs/multi{nn}/i0_s3000000/ant_20_100-v0_1/ant_20_100-v0"
    args.model_zip = f"outputs/ant_20_100-v0/logs/multi{nn}/i0_s3000000/ant_20_100-v0_1/best_model.zip"
    args.n_timesteps = int(1e3)
    args.render = True

    args.body_id = 7

    if args.body_id==-1:
        ret = []
        for i in range(20):
            args.body_id = i
            output(f"body_id: {args.body_id}",1)
            ret.append(eval_model())

        print(ret)
        print(np.mean(ret))
    else:
        output(f"body_id: {args.body_id}",1)
        eval_model(seed=3)
