import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--test-n", type=int, default=1, help="test the same setting N times with different seed.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with-bodyinfo", default=True, action="store_true")
    parser.add_argument("--train-bodies", type=str, default="0,1,2,4,100,101,102,104")
    parser.add_argument("--test-bodies", type=str, default="3,103")
    parser.add_argument("--exp", type=str, default="exp_v2", help="Experiment Name, if not given, use the folder defined in utils.folder.")
    return parser.parse_args()