import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--test-n", type=int, default=1, help="test the same setting N times with different seed.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with-bodyinfo", action="store_true")
    parser.add_argument("--train-bodies", type=str, default="0,100")
    parser.add_argument("--test-bodies", type=str, default="")
    parser.add_argument("--test-as-class", type=int, default=-1)
    parser.add_argument("--exp", type=str, default="exp_0", help="Experiment Name, if not given, use the folder defined in utils.folder.")

    parser.add_argument("--disable-wrapper", action="store_true", help="For exp_add_help")
    parser.add_argument("--with-checkpoint", action="store_true", help="save checkpoints along training.")
    return parser.parse_args()