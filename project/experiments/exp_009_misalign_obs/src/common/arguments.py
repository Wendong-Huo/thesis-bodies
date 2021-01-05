import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--train_bodies", type=str, default="", help="Format: --train_bodies=0,100,300")
    parser.add_argument("--test_bodies", type=str, default="", help="Format: --test_bodies=0,100,300")

    parser.add_argument("--test_steps", type=int, default=1000, help="total time steps for testing.")
    parser.add_argument("--train_steps", type=float, default=1e6, help="total time steps for testing.")

    parser.add_argument("--num_venvs", type=int, default=16, help="How many envs you want to vectorize (train together).")

    parser.add_argument("--with_bodyinfo", action="store_true")
    parser.add_argument("--stack_frames", type=int, default=1, help="How many frames do you want to stack for training and testing. ")
    parser.add_argument("--vec_normalize", action="store_true", help="use VecNormalize")
    parser.add_argument("--with_checkpoint", action="store_true", help="save checkpoints along training.")

    parser.add_argument("--threshold_threshold", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")
    parser.add_argument("--threshold_value", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")

    parser.add_argument("--initialize_weights_from", type=str, default="", help="better initialization from model file.")

    parser.add_argument("--model_filename", type=str, default="", help="model to test.")
    
    args = parser.parse_args()

    args.train_steps = int(args.train_steps)
    args.train_bodies_str = args.train_bodies
    args.train_bodies = str2array(args.train_bodies_str)
    args.test_bodies_str = args.test_bodies
    args.test_bodies = str2array(args.test_bodies_str)

    return args


def str2array(_str, separation=","):
    assert isinstance(_str, str)
    array = []
    if len(_str) > 0:
        array = [int(x) for x in _str.split(separation)]
    return array
