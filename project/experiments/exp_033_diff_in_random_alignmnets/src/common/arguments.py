import sys, argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--train_bodies", type=str, default="", help="Format: --train_bodies=0,100,300")
    parser.add_argument("--test_bodies", type=str, default="", help="Format: --test_bodies=0,100,300")

    parser.add_argument("--test_steps", type=int, default=1000, help="total time steps for testing.")
    parser.add_argument("--train_steps", type=float, default=2e6, help="total time steps for training.")
    parser.add_argument("--eval_steps", type=float, default=1e5, help="eval policy every # steps during training.")

    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Setting learning rate.")
    parser.add_argument("--num_venvs", type=int, default=16, help="How many envs you want to vectorize (train together).")

    parser.add_argument("--with_bodyinfo", action="store_true")
    parser.add_argument("--stack_frames", type=int, default=1, help="How many frames do you want to stack for training and testing. ")
    parser.add_argument("--with_checkpoint", action="store_true", help="save checkpoints along training.")

    parser.add_argument("--threshold_threshold", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")
    parser.add_argument("--threshold_value", type=float, default=0.0, help="activation function used in training. 0.0 is equivalent to ReLU")

    parser.add_argument("--model_filename", type=str, default="", help="model to test.")
    # parser.add_argument("--initialize_weights_from", type=str, default="", help="better initialization from model file.")

    parser.add_argument("--body_folder", type=str, default="../input_data/bodies", help="folders that contains body xml files")
    
    parser.add_argument("--topology_wrapper", type=str, default="", help="Switch for different experiments. Could be: same|diff")
    parser.add_argument("--wrapper_case", type=str, default="Walker2DHopperWrapper", help="special wrapper for different experiments.")
    parser.add_argument("--realign_method", type=str, default="", help="Only works when --topologies=same. See exp_012's hypothesis. Could be: general_only|joints_only|feetcontact_only|...")
    parser.add_argument("--custom_align_max_joints", type=int, default=10, help="For CustomAlignWrapper. 8 for Vanilla4, 10 for RandomBody.")
    parser.add_argument("--custom_alignment", type=str, default="", help="For CustomAlignWrapper. e.g. '0,1,2;0,1,2;2,0,1;1,0,2' for 4 observations with size of 3. ")
    # parser.add_argument("--misalign_obs", action="store_true", help="first misalignment test.")
    # parser.add_argument("--random_align_obs", action="store_true", help="second misalignment test.")
    # parser.add_argument("--preserve_header", action="store_true", help="preserve_header when misalign others")
    parser.add_argument("--random_even_same_body", action="store_true", help="all training bodies have different orders in observation.")
    # parser.add_argument("--preserve_feet_contact", action="store_true", help="preserve_feet_contact when misalign the rest (only obs of joints)")
    parser.add_argument("--disable_reordering", action="store_true", help="For MutantWrapper, add this to disable reordering even when use MutantWrapper. So at test time, when rendering, we can see the coloring.")

    parser.add_argument("--ga_job_id", type=int, default=-1, help="For GA, the individual id.")
    parser.add_argument("--ga_parent_id", type=int, default=-1, help="For GA, the individual id.")

    parser.add_argument("--pns", action="store_true", help="Use Mlp with PNS instead of MlpPolicy.")
    parser.add_argument("--pns_init", action="store_true", help="By default, first 8 numbers in observation is general information. Init with an I matrix in sensor weight.")
    parser.add_argument("--pns_fix_cns", action="store_true", help="By default, every parameter in policy (both cns and pns) is trainable. If we can assume the cns is good enough, we can use this flag to fix the parameter in cns and only train pns.")
    parser.add_argument("--cnspns", action="store_true", help="Use Official Version of CNSPNSPPO")
    parser.add_argument("--cnspns_sensor_channel", type=int, default=32, help="Number of sensor channels. (Ants has 28 input, so 32 might be a reasonable choice)")
    parser.add_argument("--cnspns_motor_channel", type=int, default=8, help="Number of channels. (Ants has 8 joints, so 8 might be a reasonable choice)")
    parser.add_argument("--cnspns_fix_cns", action="store_true", help="Fix the parameters in CNS, only train PNS.")

    parser.add_argument("--rl_hyperparameter", type=str, default="PPO", help="The name of hyperparameter set used for training.")


    parser.add_argument("--one_snapshot_at", type=int, default=-1, help="For save images, only save one picture at certain step.")
    parser.add_argument("--skip_solved_threshold", type=float, default=-1, help="Define a value for solved, skip training on that body until everyone pass that threshold. -1 for disabling this function.")
    parser.add_argument("-f", "--subfolder", type=str, default="default", help="Subfolder for this run.")
    parser.add_argument("--force_read", action="store_true")

    # parser.add_argument("--vec_normalize", action="store_true", default=True, help="use VecNormalize") # let's stick to true.

    if "/tests/test_" in sys.argv[0]: # hack: unittest from file, standalone
        args = parser.parse_args(sys.argv[1:])
        sys.argv = [sys.argv[0]]
    if sys.argv[0]=="python -m unittest": # hack: unittest from command line
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    args.train_steps = int(args.train_steps)
    args.train_bodies_str = args.train_bodies
    args.train_bodies = str2array(args.train_bodies_str)
    args.test_bodies_str = args.test_bodies
    args.test_bodies = str2array(args.test_bodies_str)

    args.vec_normalize = True

    return args


def str2array(_str, separation=","):
    assert isinstance(_str, str)
    array = []
    if len(_str) > 0:
        array = [int(x) for x in _str.split(separation)]
    return array
