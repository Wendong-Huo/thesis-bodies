import argparse


def get_args():
    """arguments for main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", type=int, default=2, help="Output information level: 0. nothing, 1. critical info, 2. common info, 3. debug info.")

    parser.add_argument("--exp-name", type=str, default="", help="Name of experiment.")

    # For step 1
    parser.add_argument("--num-bodies", type=int, default=20, help="Number of total different bodies you want to use in the experiments.")
    parser.add_argument("--seed-bodies", type=int, default=0, help="Seed for generating random bodies, working together with num-bodies and body-variation-range.")
    parser.add_argument("--body-variation-range", type=int, default=10, help="A percentage, maximum variation range. If set to 10, all variables will randomly scale from 0.9 to 1.1 uniformly.")
    parser.add_argument("--dataset-exist-ok", type=int, default=1, help="Overwrite bodies without warning.")
    parser.add_argument("--template-body", type=str, default="walker2d", help="The template file used for generating new bodies. Filename in the `body-templates` folder.")
    # For step 2
    parser.add_argument("--vacc", action="store_true", help="If run program on vacc, use this flag to speed up.")
    parser.add_argument("--in-parallel", action="store_true", help="We can start all experiments in parallel and it will need a huge amount of resources. For testing, you can start experiment in series, one by one.")
    parser.add_argument("-n", "--n-timesteps", type=float, default=5e6, help="Setting training timesteps for both multi-body and single-body.")
    parser.add_argument("-p", "--partition", type=str, default="short", help="Setting the partition on VACC that used to run the experiments.")
    parser.add_argument("-m", "--memory", type=str, default="6G", help="Setting the memory needed on VACC that used to run the experiments.")
    parser.add_argument("--no-single", action="store_true", default=True, help="Skip the single-body training.")
    parser.add_argument("--no-multi", action="store_true", help="Skip the multi-body training.")
    return parser.parse_args()

def get_args_train():
    """arguments for _train.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default", help="Name of experiment.")
    parser.add_argument("--exp-idx", type=int, default=0, help="Index of experiment.")

    parser.add_argument("--hyperparam", type=str, default="default", help="Which hyper parameter set is used.")
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument("--dataset", type=str, default="dataset/walker2d_10_10-v0", help="Path of dataset.")
    
    # Train on one body or multiple bodies
    parser.add_argument("--single", action="store_true", default=False, help="Train using one single body.")
    parser.add_argument("--single-idx", type=int, default=0, help="Which single body is used to train on.")
    parser.add_argument("--body-ids", type=str, default="0", help="Which bodies are used to train on, seperate by comma, e.g. --body-ids=0,1,2,3")
    parser.add_argument("--eval-ids", type=str, default="0", help="Which bodies are used to evaluate, seperate by comma, e.g. --body-ids=0,1,2,3")

    # Visualization during training
    parser.add_argument("--watch-train", action="store_true", default=False, help="Show pybullet window of the training agent during training.")
    parser.add_argument("--watch-eval", action="store_true", help="Show pybullet window of the evaluating agent during training.")

    parser.add_argument("-tb", "--tensorboard-log", type=str, default="out.tb", help="Path to tensorboard log.")
    parser.add_argument("--log-folder", type=str, default="out.logs", help="Path to log models.")
    
    parser.add_argument("--n-timesteps", type=float, default=3e5, help="Training timesteps.")
    # parser.add_argument("--eval-freq", type=float, default=1e4, help="Evaluate policy on the training body every # timesteps.") # eval right before dumping the log, no need to set.
    parser.add_argument("--log-interval", type=int, default=1, help="Write tensorboard log every # iteraction.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Every evaluation contains # episodes.")

    # This is the main difference that is testing.
    parser.add_argument("--with-bodyinfo", action="store_true", default=False, help="Train with body infomation, which means the training algorithms will concatenate params to observation.")
    return parser.parse_args()

def get_args_report():
    """arguments for _report.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="vacc_download/walker2d_10_10-v1", help="Path to stored experiment data.")
    # parser.add_argument("--folder", type=str, default="outputs/walker2d_10_10-v0", help="Path to stored experiment data.")

    parser.add_argument("--figure-1", action="store_true", default=True, help="Figure 1")
    parser.add_argument("--n-timesteps", type=float, default=1e6, help="How long an episode do we measure.")
    parser.add_argument("--force-reload", action="store_true", help="Don't use pickle cache.")
    
    return parser.parse_args()

def get_args_eval():
    """arguments for _eval.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="vacc_download/walker2d_10_10-v1", help="Path to stored experiment data.")
    return parser.parse_args()


# args = get_args()
# args_train = get_args_train()
