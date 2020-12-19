import time, pickle, os
import numpy as np
import utils
import wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize, util

def fname(train):
    if isinstance(train, list):
        fname = f"{train[0]}-{train[1]}"
    else:
        fname = f"single-{train}"
    return fname

def test(seed, model_filename, vec_filename, train, test, body_info=0, render=False):
    print("Testing:")
    print(f" Seed {seed}, model {model_filename} vec {vec_filename}")
    print(f" Train on {train}, test on {test}, w/ bodyinfo {body_info}")
    eval_env = utils.make_env(render=render, robot_body=test, body_info=body_info)
    eval_env = DummyVecEnv([eval_env])
    eval_env = VecNormalize.load(vec_filename, eval_env)
    eval_env.norm_reward = False

    eval_env.seed(seed)
    model = PPO.load(model_filename)

    obs = eval_env.reset()
    distance_x = 0
    # print(obs)
    total_reward = 0
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        else: # the last observation will be after reset, so skip the last
            distance_x = eval_env.envs[0].robot.body_xyz[0]
        total_reward += reward[0]
        if render:
            time.sleep(0.01)
        
    eval_env.close()
    print(f"train {train}, test {test}, body_info {body_info}, step {step}, total_reward {total_reward}, distance_x {distance_x}")
    return total_reward, distance_x

def test_all(N=20):
    folder = utils.folder
    data = {
        "reward": {
            "train_on_one": {},
            "train_on_two": {},
            "train_on_two_with_bodyinfo_zero": {},
            "train_on_two_with_bodyinfo_one": {},
            "train_on_two_with_bodyinfo_two": {},
        },
        "distance": {
            "train_on_one": {},
            "train_on_two": {},
            "train_on_two_with_bodyinfo_zero": {},
            "train_on_two_with_bodyinfo_one": {},
            "train_on_two_with_bodyinfo_two": {},
        },
    }

    for trainon in [0,1,2]:
        model_filename = f"{folder}/model-ant-{fname(trainon)}.zip"
        vec_filename = f"{folder}/model-ant-{fname(trainon)}-vecnormalize.pkl"
        if os.path.exists(model_filename):
            for teston in [0,1,2]:
                rs = []
                ds = []
                for i in range(N):
                    r, d = test(seed=i, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=teston)
                    rs.append(r)
                    ds.append(d)
                data["reward"]["train_on_one"][f"train-{trainon}-test-{teston}"] = rs
                data["distance"]["train_on_one"][f"train-{trainon}-test-{teston}"] = ds

    for trainon in [[0,1],[0,2],[1,2]]:
        model_filename = f"{folder}/model-ant-{fname(trainon)}.zip"
        vec_filename = f"{folder}/model-ant-{fname(trainon)}-vecnormalize.pkl"
        if os.path.exists(model_filename):
            for teston in [0,1,2]:
                rs = []
                ds = []
                for i in range(N):
                    r, d = test(seed=i, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=teston)
                    rs.append(r)
                    ds.append(d)
                data["reward"]["train_on_two"][f"train-{fname(trainon)}-test-{teston}"] = rs
                data["distance"]["train_on_two"][f"train-{fname(trainon)}-test-{teston}"] = ds

    for trainon in [[0,1],[0,2],[1,2]]:
        model_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo.zip"
        vec_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo-vecnormalize.pkl"
        if os.path.exists(model_filename):
            for teston in [0,1,2]:
                rs = []
                ds = []
                for i in range(N):
                    r, d = test(seed=i, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=teston, body_info=0)
                    rs.append(r)
                    ds.append(d)
                data["reward"]["train_on_two_with_bodyinfo_zero"][f"train-{fname(trainon)}-test-{teston}"] = rs
                data["distance"]["train_on_two_with_bodyinfo_zero"][f"train-{fname(trainon)}-test-{teston}"] = ds

    for trainon in [[0,1],[0,2],[1,2]]:
        model_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo.zip"
        vec_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo-vecnormalize.pkl"
        if os.path.exists(model_filename):
            for teston in [0,1,2]:
                rs = []
                ds = []
                for i in range(N):
                    r, d = test(seed=i, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=teston, body_info=1)
                    rs.append(r)
                    ds.append(d)
                data["reward"]["train_on_two_with_bodyinfo_one"][f"train-{fname(trainon)}-test-{teston}"] = rs
                data["distance"]["train_on_two_with_bodyinfo_one"][f"train-{fname(trainon)}-test-{teston}"] = ds

    for trainon in [[0,1],[0,2],[1,2]]:
        model_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo.zip"
        vec_filename = f"{folder}/model-ant-{fname(trainon)}-with-bodyinfo-vecnormalize.pkl"
        if os.path.exists(model_filename):
            for teston in [0,1,2]:
                rs = []
                ds = []
                for i in range(N):
                    r, d = test(seed=i, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=teston, body_info=2)
                    rs.append(r)
                    ds.append(d)
                data["reward"]["train_on_two_with_bodyinfo_two"][f"train-{fname(trainon)}-test-{teston}"] = rs
                data["distance"]["train_on_two_with_bodyinfo_two"][f"train-{fname(trainon)}-test-{teston}"] = ds


    with open(f"{folder}/data-simple.pickle", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":  # noqa: C901
    test_all(N=20)
