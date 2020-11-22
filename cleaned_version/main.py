# Author: Sida Liu, 2020
#   Starting experiments
#
# for all available arguments, refer to arguments.py
from arguments import get_args
import step1_generate_bodies
import step2_train_on_one_body

args = get_args()


def main():
    env_id, dataset_path = step1_generate_bodies.generate_bodies()
    step2_train_on_one_body.train(env_id, dataset_path)
    pass


if __name__ == "__main__":
    main()
