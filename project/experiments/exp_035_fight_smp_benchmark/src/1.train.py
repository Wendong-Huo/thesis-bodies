import re
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.save_util import load_from_zip_file

from common import wrapper_custom_align, wrapper_diff, wrapper_mut, wrapper_pns

import common.common as common
from common.utils import load_parameters_from_path
import common.wrapper as wrapper
import common.gym_interface as gym_interface
import common.callbacks as callbacks
from common.activation_fn import MyThreshold
from common.pns import PNSPPO, PNSMlpPolicy
from common.cnspns import CNSPNSPPO, CNSPNSPolicy

if __name__ == "__main__":

    args = common.args
    print(args)

    args.test_bodies = args.train_bodies # omit test_bodies in command from now on.
    args.initialize_weights_from = args.model_filename

    # SAC.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(common.seed)

    saved_model_filename = common.build_model_filename(args)

    hyperparams = common.load_hyperparameters(conf_name=args.rl_hyperparameter)
    print(hyperparams)

    # Make every env has the same obs space and action space
    default_wrapper = []
    # if padding zero:
    #   default_wrapper.append(wrapper.WalkerWrapper)

    if args.topology_wrapper == "same":
        body_type = 0
        for body in args.train_bodies + args.test_bodies:
            if body_type == 0:
                body_type = body//100
            else:
                assert body_type == body//100, "Training on different body types."
        if args.realign_method != "":
            default_wrapper.append(wrapper.ReAlignedWrapper)
    elif args.topology_wrapper == "diff":
        default_wrapper.append(wrapper_diff.get_wrapper_class())
    elif args.topology_wrapper == "MutantWrapper":
        default_wrapper.append(wrapper_mut.MutantWrapper)
    elif args.topology_wrapper == "CustomAlignWrapper":
        default_wrapper.append(wrapper_custom_align.CustomAlignWrapper)
    else:
        pass  # no need for wrapper

    # if args.cnspns:
    # hard code for now. could be automatically determined.
    _w = wrapper_pns.make_same_dim_wrapper(obs_dim=28, action_dim=8)
    default_wrapper.append(_w)

    assert len(args.train_bodies) > 0, "No body to train."
    if args.with_bodyinfo:
        default_wrapper.append(wrapper.BodyinfoWrapper)

    print("Making train environments...")
    venv = DummyVecEnv([gym_interface.make_env(rank=i, seed=common.seed, wrappers=default_wrapper, render=args.render,
                                               robot_body=args.train_bodies[i % len(args.train_bodies)],
                                               dataset_folder=args.body_folder) for i in range(args.num_venvs)])

    normalize_kwargs = {}
    if args.vec_normalize:
        normalize_kwargs["gamma"] = hyperparams["gamma"]
        if len(args.model_filename) > 0:
            venv = VecNormalize.load(common.get_vec_pkl_from_model_filename(args.model_filename), venv)
        else:
            venv = VecNormalize(venv, **normalize_kwargs)

    if args.stack_frames > 1:
        venv = VecFrameStack(venv, args.stack_frames)

    print("Making eval environments...")
    all_callbacks = []
    for rank_idx, test_body in enumerate(args.test_bodies):
        body_info = 0
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=rank_idx, seed=common.seed+1, wrappers=default_wrapper, render=False,
                                                        robot_body=test_body, body_info=body_info,
                                                        dataset_folder=args.body_folder)])
        if args.vec_normalize:
            eval_venv = VecNormalize(eval_venv, norm_reward=False, **normalize_kwargs)
        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)
        eval_callback = callbacks.EvalCallback_with_prefix(
            eval_env=eval_venv,
            best_model_save_path=str(common.output_data_folder/"models"/common.args.subfolder/saved_model_filename),
            prefix=f"{test_body}",
            n_eval_episodes=3,
            eval_freq=int(args.eval_steps/args.num_venvs),  # will implicitly multiplied by (train_num_envs)
            deterministic=True,
        )
        all_callbacks.append(eval_callback)

    if args.with_checkpoint:
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=str(common.output_data_folder/'checkpoints'), name_prefix=args.train_bodies)
        all_callbacks.append(checkpoint_callback)
        if args.vec_normalize:
            save_vec_callback = callbacks.SaveVecNormalizeCallback(save_freq=1000, save_path=str(
                common.output_data_folder/'checkpoints'), name_prefix=args.train_bodies)
            all_callbacks.append(save_vec_callback)

    if args.skip_solved_threshold > 0:
        skip_solved_callback = callbacks.SkipSolvedCallback(args.skip_solved_threshold)
        all_callbacks.append(skip_solved_callback)

    all_callbacks.append(callbacks.InspectionCallback())

    if args.pns:
        model_cls = PNSPPO
        policy_cls = PNSMlpPolicy
    elif args.cnspns:
        model_cls = CNSPNSPPO
        policy_cls = CNSPNSPolicy
    else:
        model_cls = PPO
        policy_cls = "MlpPolicy"

    hyperparams = common.clean_hyperparams_before_run(hyperparams)
    th.manual_seed(args.seed)
    model = model_cls(policy_cls, venv, verbose=1, tensorboard_log=str(
        common.output_data_folder / "tensorboard" / common.args.subfolder / saved_model_filename),
        seed=common.seed, **hyperparams)

    if args.debug:
        if False: # write params to disk
            to_save = []
            for p in model.policy.parameters():
                if not hasattr(p, "robot_id"):
                    p = p.detach().cpu().numpy()
                    to_save.append(p)
            import pickle
            with open("output_data/tmp/initialization.pkl", "wb") as f:
                print(len(to_save))
                pickle.dump(to_save, f)
        common.inspect_regular_parameters(model.policy, "Before load initialization from disk.")
        import pickle
        with open("output_data/tmp/initialization.pkl", "rb") as f:
            to_load = pickle.load(f)
        with th.no_grad():
            for p, lp in zip(model.policy.parameters(), to_load):
                if not hasattr(p, "robot_id"):
                    p.copy_(th.Tensor(lp))
        common.inspect_regular_parameters(model.policy, "After load initialization from disk.")

    if len(args.model_filename) > 0:
        model = load_parameters_from_path(model=model, model_filename=args.model_filename,
                            model_cls=model_cls, bodies=args.train_bodies, default_wrapper=default_wrapper)

    if args.cnspns_fix_cns:
        for parameter in model.policy.parameters():
            if hasattr(parameter, "pns_type"):
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    if args.cnspns_start_with_identity:
        for parameter in model.policy.parameters():
            if hasattr(parameter, "pns_type"):
                with th.no_grad():
                    if len(parameter.shape)==2:
                        assert parameter.shape[0] == parameter.shape[1], "Nets are not square, so it can't be identity matrix"
                        parameter.copy_(th.Tensor(th.eye(n=parameter.shape[0])))
                    elif len(parameter.shape)==1:
                        parameter.copy_(th.zeros_like(parameter))

    for parameter in model.policy.parameters():
        if hasattr(parameter, "robot_id"):
            if int(parameter.robot_id) in args.cnspns_fix_pns_bodies:
                parameter.requires_grad = False

    if args.debug:
        th.manual_seed(args.seed)
        model.set_random_seed(args.seed)
        venv.seed(args.seed)
    try:
        model.learn(total_timesteps=args.train_steps, callback=all_callbacks)
    except KeyboardInterrupt:
        pass
    model.save(str(common.output_data_folder/"models"/common.args.subfolder/saved_model_filename))

    if args.vec_normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(str(common.output_data_folder/"models"/common.args.subfolder/f"{saved_model_filename}.vnorm.pkl"))

    venv.close()
