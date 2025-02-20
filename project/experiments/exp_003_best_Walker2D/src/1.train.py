from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

import common.common as common
import common.wrapper as wrapper
import common.gym_interface as gym_interface
import common.callbacks as callbacks
from common.activation_fn import MyThreshold

if __name__ == "__main__":

    args = common.args
    print(args)

    # SAC.learn need this. If use SubprocVecEnv instead of DummyVecEnv, you need to seed in each subprocess.
    set_random_seed(common.seed)

    saved_model_filename = common.build_model_filename(args)

    hyperparams = common.load_hyperparameters(conf_name="SAC")
    print(hyperparams)

    # Make every env has the same obs space and action space
    default_wrapper = [wrapper.WalkerWrapper]

    assert len(args.train_bodies) > 0, "No body to train."
    if args.with_bodyinfo:
        default_wrapper += [wrapper.BodyinfoWrapper]
    venv = DummyVecEnv([gym_interface.make_env(rank=i, seed=common.seed, wrappers=default_wrapper, render=args.render,
                                               robot_body=args.train_bodies[i % len(args.train_bodies)]) for i in range(args.num_venvs)])

    normalize_kwargs = {}
    if args.vec_normalize:
        normalize_kwargs["gamma"] = hyperparams["gamma"]
        venv = VecNormalize(venv, **normalize_kwargs)

    if args.stack_frames > 1:
        venv = VecFrameStack(venv, args.stack_frames)

    keys_remove = ["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        if key in hyperparams:
            del hyperparams[key]

    all_callbacks = []
    for test_body in args.test_bodies:
        body_info = 0
        eval_venv = DummyVecEnv([gym_interface.make_env(rank=0, seed=common.seed+1, wrappers=default_wrapper, render=False,
                                                        robot_body=test_body, body_info=body_info)])
        if args.vec_normalize:
            eval_venv = VecNormalize(eval_venv, norm_reward=False, **normalize_kwargs)
        if args.stack_frames > 1:
            eval_venv = VecFrameStack(eval_venv, args.stack_frames)
        eval_callback = callbacks.EvalCallback_with_prefix(
            eval_env=eval_venv,
            best_model_save_path=str(common.output_data_folder/"models"/saved_model_filename),
            prefix=f"{test_body}",
            n_eval_episodes=3,
            eval_freq=1e3,  # will implicitly multiplied by (train_num_envs)
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

    hyperparams['policy_kwargs']['activation_fn'] = MyThreshold

    model = SAC('MlpPolicy', venv, verbose=1, tensorboard_log=str(common.output_data_folder/"tensorboard"/saved_model_filename), seed=common.seed, **hyperparams)

    if len(args.initialize_weights_from) > 0:
        try:
            load_model = SAC.load(args.initialize_weights_from)
            load_weights = load_model.policy.state_dict()
            model.policy.load_state_dict(load_weights)
            print(f"Weights loaded from {args.initialize_weights_from}")
        except Exception:
            print("Initialize weights error.")
            raise Exception

    try:
        model.learn(total_timesteps=args.train_steps, callback=all_callbacks)
    except KeyboardInterrupt:
        pass
    model.save(str(common.output_data_folder/"models"/saved_model_filename))

    if args.vec_normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(str(common.output_data_folder/"models"/f"{saved_model_filename}.vnorm.pkl"))

    venv.close()
