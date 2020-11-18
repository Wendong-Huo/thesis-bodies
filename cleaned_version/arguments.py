import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-level", type=int, default=3, help="Output information level: 0. nothing, 1. critical info, 2. common info, 3. debug info.")
    parser.add_argument("--num-bodies", type=int, default=10, help="Number of total different bodies you want to use in the experiments.")
    parser.add_argument("--seed-bodies", type=int, default=0, help="Seed for generating random bodies.")
    parser.add_argument("--name-bodies", type=str, default="Walker2D", help="Name for generated bodies. It will be the name in the `dataset` folder.")
    parser.add_argument("--name-bodies-exist-ok", type=int, default=1, help="Overwrite bodies without warning.")
    parser.add_argument("--template-body", type=str, default="walker2d", help="The template file used for generating new bodies. Filename in the `body-templates` folder.")
    return parser.parse_args()


args = get_args()
