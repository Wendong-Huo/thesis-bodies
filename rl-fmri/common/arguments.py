import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--test-n", type=int, default=1, help="test the same setting N times with different seed.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--with-bodyinfo", action="store_true")
    parser.add_argument("--train-bodies", type=str, default="0,100")
    parser.add_argument("--test-bodies", type=str, default="0")
    parser.add_argument("--test-as-class", type=int, default=-1)
    parser.add_argument("--exp", type=str, default="exp_1", help="Experiment Name, if not given, use the folder defined in utils.folder.")

    parser.add_argument("--disable-wrapper", default=True, action="store_true", help="For exp_add_help")
    parser.add_argument("--with-checkpoint", action="store_true", help="save checkpoints along training.")

    parser.add_argument("--save-fmri", action="store_true", help="save fMRI data.")
    parser.add_argument("--save-obs-data", action="store_true", help="save obs data during testing.")
    parser.add_argument("--compare-seed", action="store_true", help="add another experiment with another seed.")
    parser.add_argument("--test-steps", type=int, default=10, help="total time steps for testing.")
    parser.add_argument("--train-steps", type=float, default=1e6, help="total time steps for testing.")

    parser.add_argument("--initialize-weights-from", type=str, default="", help="better initialization from model file.")

    parser.add_argument("--vec-normalize", action="store_true", help="use VecNormalize")
    parser.add_argument("--disable-saving-image", action="store_true", help="")
    parser.add_argument("--num-venvs", type=int, default=16, help="How many envs you want to vectorize (train together).")
    

    parser.add_argument("--threshold_threshold", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")
    parser.add_argument("--threshold_value", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")
    parser.add_argument("--stack_frames", type=int, default=4, help="How many frames do you want to stack for training and testing. ")

    # obs_predict.py: Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--exclude-z', action="store_true")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000000, metavar='N',
    #                     help='input batch size for testing (default: 1000000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')

    return parser.parse_args()
