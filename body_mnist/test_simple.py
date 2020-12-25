import time
import pickle
import os
import numpy as np
import utils
import wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize, util


def fname(train):
    if isinstance(train, list):
        fname = '-'.join(str(x) for x in train)
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
    if render:
        eval_env.env_method("set_view")
    distance_x = 0
    # print(obs)
    total_reward = 0
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done:
            break
        else:  # the last observation will be after reset, so skip the last
            distance_x = eval_env.envs[0].robot.body_xyz[0]
        total_reward += reward[0]
        if render:
            time.sleep(0.01)

    eval_env.close()
    print(f"train {train}, test {test}, body_info {body_info}, step {step}, total_reward {total_reward}, distance_x {distance_x}")
    return total_reward, distance_x


def test_all():
    folder = utils.folder
    bodyinfos = [0, 1]
    max_num_trainedon = 10
    N = utils.args.test_n

    import glob
    import re
    files = glob.glob(f"{folder}/*.zip")
    all_possible_bodies = set()
    trainons = []
    for file in files:
        if "-with-bodyinfo" in file:
            continue
        ids = r"[0-9]+"
        _ids = ids
        for i in range(max_num_trainedon):
            match = re.search(r'ant-('+_ids+r')\.zip', file)
            if match:
                match_result = match[1]
                # print(file, match_result)
                trainon = match_result.split("-")
                trainon = [int(x) for x in trainon]
                for x in trainon:
                    all_possible_bodies.add(x)
                trainons.append(trainon)
            _ids = _ids + "-" + ids

    print(trainons)
    print(all_possible_bodies)

    n = 0

    testons = {}
    for trainon in trainons:
        if len(trainon) != 8:
            continue
        str_trainon = '-'.join(str(x) for x in trainon)
        testons[str_trainon] = []
        for testable in all_possible_bodies:
            if testable in trainon:
                continue
            testons[str_trainon].append(testable)
    print(testons)

    all_testons = {}
    for trainon in trainons:
        str_trainon = '-'.join(str(x) for x in trainon)
        if len(trainon)==1:
            all_testons[str_trainon] = trainon
        else:
            for key in testons:
                if str_trainon in key:
                    all_testons[str_trainon] = testons[key]

    print(all_testons)

    # with open(f"{folder}/test-results.pickle", "rb") as f:
    #     test_results = pickle.load(f)

    test_results = {}
    test_results["reward"] = {}
    test_results["distance"] = {}
    for with_bodyinfo in range(2):
        if with_bodyinfo:
            str_with_bodyinfo = "-with-bodyinfo"
        else:
            str_with_bodyinfo = ""
            # continue

        test_results["reward"][with_bodyinfo] = {}
        test_results["distance"][with_bodyinfo] = {}

        for trainon in trainons:
            if len(trainon) == 1 and with_bodyinfo == 1:
                continue
            if len(trainon) == 4 and with_bodyinfo == 1:
                continue

            str_trainon = '-'.join(str(x) for x in trainon)
            teston = all_testons[str_trainon]

            test_results["reward"][with_bodyinfo][str_trainon] = {}
            test_results["distance"][with_bodyinfo][str_trainon] = {}
            model_filename = f"{folder}/model-ant-{fname(trainon)}{str_with_bodyinfo}.zip"
            vec_filename = model_filename[:-4] + "-vecnormalize.pkl"
            for test_body in teston:
                if test_body in trainon and len(trainon) > 1:
                    continue
                if test_body not in trainon and len(trainon) == 1:
                    continue
                test_results["reward"][with_bodyinfo][str_trainon][test_body] = {}
                test_results["distance"][with_bodyinfo][str_trainon][test_body] = {}
                for bodyinfo in bodyinfos:
                    if len(trainon) == 1 and bodyinfo == 1:
                        continue
                    if with_bodyinfo == 0 and bodyinfo == 1:
                        continue
                    test_results["reward"][with_bodyinfo][str_trainon][test_body][bodyinfo] = []
                    test_results["distance"][with_bodyinfo][str_trainon][test_body][bodyinfo] = []
                    for s in range(N):
                        # reward, distance = 0, 0
                        reward, distance  = test(seed=s, model_filename=model_filename, vec_filename=vec_filename, train=trainon, test=test_body, body_info=bodyinfo, render=False)
                        n += 1
                        test_results["reward"][with_bodyinfo][str_trainon][test_body][bodyinfo].append(reward)
                        test_results["distance"][with_bodyinfo][str_trainon][test_body][bodyinfo].append(distance)

    with open(f"{folder}/test-results.pickle", "wb") as f:
        pickle.dump(test_results, f)

    import pprint
    pprint.pprint(test_results)
    print(f"total n={n}")
    # pprint.pprint(sorted(trainons))


if __name__ == "__main__":  # noqa: C901
    args = utils.args
    folder = utils.folder
    
    # test_all()
    train_bodies = [int(x) for x in args.train_bodies.split(',')]
    test_bodies = [int(x) for x in args.test_bodies.split(',')]
    for train_body in train_bodies:
        for test_body in test_bodies:
            model_filename = f"{folder}/model-ant-{train_body}.zip"
            vec_filename = model_filename[:-4] + "-vecnormalize.pkl"
            test(1, model_filename, vec_filename, [], test_body, body_info=0, render=args.render)