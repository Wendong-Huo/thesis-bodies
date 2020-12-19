import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--body-id", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with-bodyinfo", action="store_true")
    parser.add_argument("--train-on-both-bodies", action="store_true")
    parser.add_argument("--train-bodies", nargs="+", type=int, default=[0,1])
    parser.add_argument("--exp", type=str, default="", help="Experiment Name, if not given, use the folder defined in utils.folder.")
    return parser.parse_args()