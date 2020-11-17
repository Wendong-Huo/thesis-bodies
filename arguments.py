import argparse
from utils import ALGOS
from utils.utils import StoreDict


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("--hyperparameters", type=str, default="Walker2DBulletEnv-v0", help="")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="tb", type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory in the replay buffer will be truncated after reloading the replay buffer.", default=True,
        type=bool,)
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument("--eval-freq", help="Evaluate the agent every n steps (if negative, no evaluation)", default=10000, type=int)
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument("--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False)
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--n-trials", help="Number of trials for optimizing hyperparameters", type=int, default=10)
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search")
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument("--sampler", help="Sampler to use when optimizing hyperparameters", type=str, default="tpe", choices=["random", "tpe", "skopt"],)
    parser.add_argument("--pruner", help="Pruner to use when optimizing hyperparameters", type=str, default="median", choices=["halving", "median", "none"],)
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument("--n-evaluations", help="Number of evaluations for hyperparameter optimization", type=int, default=20)
    parser.add_argument("--storage", help="Database storage path if distributed optimization should be used", type=str, default=None)
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--gym-packages", type=str, nargs="+", default=[],
                        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)")
    parser.add_argument("--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)")
    parser.add_argument("--watch-train", action="store_true", default=False)
    parser.add_argument("--watch-eval", action="store_true", default=False)
    parser.add_argument("--powercoeff", type=float, nargs=3, default=[1.,1.,1.], help="Only useful for adjusting powercoeff. Default is [1 1 1].")
    parser.add_argument("--single-idx", type=int, default=-1, help="The body id that will be trained.")
    parser.add_argument("--single-group", type=int, default=-1, help="The group of body that will be trained.")
    parser.add_argument("--dataset", type=str, default="dataset/walker2d_v6", help="Path to dataset")
    return parser.parse_args()

def get_dissect_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model-path", type=str, default="logs/malicious_info/ppo/Walker2Ds-v0_4/best_model.zip", help="Path to model.zip")
    return parser.parse_args()
