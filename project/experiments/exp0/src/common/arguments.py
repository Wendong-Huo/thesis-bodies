import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    return parser.parse_args()
