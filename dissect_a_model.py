import arguments
from utils import ALGOS
if __name__ == "__main__":
    args = arguments.get_dissect_arguments()
    algo = "ppo"
    model = ALGOS[algo].load(args.model_path, env=None)
    print(model)
