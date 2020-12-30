#--exp, --seed, --train-bodies, --test-bodies  --with-bodyinfo
import os
from torch._C import ErrorReport
import yaml
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from common.callbacks.eval import EvalCallback_with_prefix
from common.callbacks.fromzoo import SaveVecNormalizeCallback
import common.utils as utils
import common.wrapper as wrapper
from common.activation_fn import MyThreshold

if __name__ == "__main__":  # noqa: C901
    folder = utils.folder
    os.makedirs(folder, exist_ok=True)

    hyperparams = utils.load_hyperparameters()

    normalize_kwargs = {}
    normalize_kwargs["gamma"] = hyperparams["gamma"]
    
    args = utils.args
    
    # PPO.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(utils.seed)

    debug = args.debug
    with_bodyinfo = args.with_bodyinfo
    train_num_envs = args.num_venvs if not debug else 2
    total_timesteps = int(args.train_steps) if not debug else 1

    training_bodies = [int(x) for x in args.train_bodies.split(",")]
    str_ids = "-".join(str(x) for x in training_bodies)
    if args.test_bodies=="":
        test_bodies = []
    else:
        test_bodies = [int(x) for x in args.test_bodies.split(",")]
    
    # default_wrapper = wrapper.BodyinfoWrapper
    # if args.disable_wrapper:
    #     default_wrapper = None
    default_wrapper = wrapper.WalkerWrapper
    # default_wrapper = None

    if with_bodyinfo:
        env = DummyVecEnv([utils.make_env(template=utils.template(training_bodies[i%len(training_bodies)]), rank=i, seed=utils.seed, wrapper=default_wrapper, render=args.render, robot_body=training_bodies[i%len(training_bodies)], body_info=training_bodies[i%len(training_bodies)]//100) for i in range(train_num_envs)])
        save_filename = f"model-ant-{str_ids}-with-bodyinfo"
    else:
        env = DummyVecEnv([utils.make_env(template=utils.template(training_bodies[i%len(training_bodies)]), rank=i, seed=utils.seed, wrapper=default_wrapper, render=args.render, robot_body=training_bodies[i%len(training_bodies)], body_info=0) for i in range(train_num_envs)])
        save_filename = f"model-ant-{str_ids}"

    if args.vec_normalize:
        env = VecNormalize(env, **normalize_kwargs)

    if args.stack_frames>1:
        env = VecFrameStack(env, args.stack_frames)


    keys_remove =["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        del hyperparams[key]

    all_callbacks = []
    for test_body in test_bodies:
        if with_bodyinfo:
            if args.test_as_class<0:
                body_info = test_body//100
            else:
                body_info = args.test_as_class
        else:
            body_info = 0
        eval_env = DummyVecEnv([utils.make_env(template=utils.template(test_body), rank=0, seed=utils.seed+1, wrapper=default_wrapper, render=False, robot_body=test_body, body_info=body_info)])
        if args.vec_normalize:
            eval_env = VecNormalize(eval_env, norm_reward=False, **normalize_kwargs)
        if args.stack_frames>1:
            eval_env = VecFrameStack(eval_env, args.stack_frames)
        eval_callback = EvalCallback_with_prefix(
            eval_env=eval_env,
            prefix=f"{test_body}",
            n_eval_episodes=3,
            eval_freq=1e3, # will implicitly multiplied by (train_num_envs)
            deterministic=True,
        )
        all_callbacks.append(eval_callback)

    if args.with_checkpoint:
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f'{folder}/checkpoints/', name_prefix=args.train_bodies)
        all_callbacks.append(checkpoint_callback)
        if args.vec_normalize:
            save_vec_callback = SaveVecNormalizeCallback(save_freq=1000, save_path=f"{folder}/checkpoints/", name_prefix=args.train_bodies)
            all_callbacks.append(save_vec_callback)

    print(save_filename)
    print(f"Observation space: {env.observation_space.shape}")

    hyperparams['policy_kwargs']['activation_fn'] = MyThreshold

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"{folder}/tb/{save_filename}-s{utils.seed}-f{args.stack_frames}", seed=utils.seed, **hyperparams)

    if len(args.initialize_weights_from) > 0:
        try:
            load_model = PPO.load(args.initialize_weights_from, env)
            load_weights = load_model.policy.state_dict()
            model.policy.load_state_dict(load_weights)
            print(f"Weights loaded from {args.initialize_weights_from}")
        except Exception:
            print("Initialize weights error.")
            raise Exception

    try:
        model.learn(total_timesteps=total_timesteps, callback=all_callbacks)
    except KeyboardInterrupt:
        pass
    model.save(f"{folder}/{save_filename}-s{utils.seed}-f{args.stack_frames}")
    if args.vec_normalize:
    # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(f"{folder}/{save_filename}-vecnormalize.pkl")

    env.close()
