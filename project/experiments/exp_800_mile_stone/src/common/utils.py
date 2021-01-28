import os, re
import hashlib
import pathlib
import yaml

from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv
import common.gym_interface as gym_interface

from common import common

def get_exp_name():
    """Return current experiment folder, such as exp0."""
    _full_path = str(pathlib.Path().absolute())
    _paths = _full_path.split('/')
    assert _paths[-1] == "src", f"Project structure has been changed. utils.get_exp_folder() should be changed accordingly.\n{_paths}"
    assert len(_paths) > 2, "Why path is so short?"
    _folder_name = _paths[-2]
    return _folder_name


def get_output_data_folder(init=False):
    # Create output folder is not exist yet.
    _path = pathlib.Path("../../../output_data")
    if not _path.is_dir():
        print("Starting a new project? Congratulations! \n\nCreating output data path for the first time.")
        print(f"mkdir {_path.resolve()}")
        _path.mkdir()
    output_data_folder = _path / get_exp_name()
    output_data_folder.mkdir(exist_ok=True)
    _subs = ["tensorboard", "plots", "models", "saved_images", "videos", "checkpoints", "tmp", "bodies", "database", "cache", "jobs"]
    for _sub in _subs:
        (output_data_folder / _sub).mkdir(exist_ok=True)

    if init:
        # Create a symlink to output_data
        _sym_link = pathlib.Path("output_data")
        if _sym_link.is_symlink():
            _sym_link.unlink()
        _sym_link.symlink_to(output_data_folder, target_is_directory=True)

    return output_data_folder


def get_input_data_folder():
    _path = pathlib.Path("../input_data")
    assert _path.exists()
    return _path

def get_current_folder():
    return pathlib.Path()

def check_exp_folder():
    """Make sure .exp_folder contains the right folder name"""
    _exp_folder = pathlib.Path(".exp_folder")

    _folder = get_exp_name()

    if _exp_folder.exists():
        _str = _exp_folder.read_text()
        if _folder == _str:
            return
    _exp_folder.write_text(_folder)
    return


def build_model_filename(args):
    filename = "model-"
    filename += args.train_bodies_str.replace(",", "-")
    if args.with_bodyinfo:
        filename += "-body"
    # if args.vec_normalize: # vec_normalize by default
    #     filename += "-vnorm"
    if args.stack_frames>1:
        filename += f"-stack{args.stack_frames}"
    if args.threshold_threshold!=0:
        filename += f"-thr{args.threshold_threshold}"
    if len(args.initialize_weights_from) > 0:
        filename += f"-initw"
    
    if args.topology_wrapper=="same":
        if args.realign_method!="":
            filename += f"-realign{args.realign_method}"
    elif args.topology_wrapper=="diff":
        if args.wrapper_case!="":
            filename += f"-case{args.wrapper_case}"
    elif args.topology_wrapper=="MutantWrapper":
        filename += "-MutantWrapper"
    elif args.topology_wrapper=="CustomAlignWrapper":
        str2hash = args.custom_alignment
        md5_string = hashlib.md5(str2hash.encode()).hexdigest()
        filename += f"-CustomAlignWrapper-md{md5_string}"
    else:
        pass
    
    if args.pns:
        filename += "-pns"
        if args.pns_init:
            filename += "-pns_init"

    # if args.misalign_obs:
    #     filename += f"-mis"
    # if args.random_align_obs:
    #     filename += f"-ra"
    # if args.preserve_header:
    #     filename += f"-ph"
    # if args.random_even_same_body:
    #     filename += f"-resb"
    # if args.preserve_feet_contact:
    #     filename += f"-pfc"

    filename += f"-sd{args.seed}"
    return filename

def mean_and_error(_data):
    """A helper for creating error bar"""
    import numpy as np
    _data = np.array(_data)
    _two_sigma = 2*np.std(_data)
    _mean = np.mean(_data)
    print(f"{_mean:.0f} +- {_two_sigma:.0f}")
    return _mean, _two_sigma


def linux_fullscreen():
    """A helper for entering Full Screen mode in Linux.
    Faking a mouse click and a key press (Ctrl+F11).
    """
    from pymouse import PyMouse
    from pykeyboard import PyKeyboard
    m = PyMouse()
    k = PyKeyboard()
    x_dim, y_dim = m.screen_size()
    m.click(int(x_dim/3), int(y_dim/2), 1)
    k.press_key(k.control_key)
    k.tap_key(k.function_keys[11])
    k.release_key(k.control_key)

def load_hyperparameters(conf_name="MyWalkerEnv"):
    import torch.nn as nn
    with (get_input_data_folder() / "hyperparameters.yml").open() as f:
        hp = yaml.load(f, Loader=yaml.SafeLoader)
    hyperparams = hp[conf_name]
    hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    # Overwrite learning_rate using args:
    hyperparams["learning_rate"] = common.args.learning_rate
    # Use MyThreshold instead of ReLU
    # hyperparams['policy_kwargs']['activation_fn'] = MyThreshold
    return hyperparams

def clean_hyperparams_before_run(hyperparams):
    # ignore these keys
    keys_remove = ["normalize", "n_envs", "n_timesteps", "policy"]
    for key in keys_remove:
        if key in hyperparams:
            del hyperparams[key]
    return hyperparams
        

def md5(str2hash):
    return hashlib.md5(str2hash.encode()).hexdigest()

def shell_header(exp_name="Unknown", description="Unknown"):
    print(f"""#!/bin/sh
set -x
python 0.init.py
EXP_FOLDER=$(cat .exp_folder)
submit_to=submit-short.sh
# ========================================

exp_name="{exp_name}"

description="{description}"

""")

def shell_tail():
    print("""
# ========================================
# log
echo "================" >> ~/gpfs2/experiments.log
date >> ~/gpfs2/experiments.log
pwd >> ~/gpfs2/experiments.log
echo $0 >> ~/gpfs2/experiments.log
echo $exp_name >> ~/gpfs2/experiments.log
echo $description >> ~/gpfs2/experiments.log
squeue -O "JobID,Partition,Name,Nodelist,TimeUsed,UserName,StartTime,Schednodes,Command" --user=sliu1 -n $exp_name | head -n 10 >> ~/gpfs2/experiments.log
echo "================" >> ~/gpfs2/experiments.log
4-show-experiment-log.sh

""")

def get_vec_pkl_from_model_filename(model_filename):
    assert model_filename.endswith(".zip")
    return f"{model_filename[:-4]}.vecnormalize.pkl"

def load_parameters_from_path(model, model_filename, model_cls, bodies, default_wrapper):

    args = common.args
    data, params, pytorch_variables = load_from_zip_file(model_filename)
    robot_ids_in_file = []
    if args.cnspns:
        for parameter_name, module in params['policy'].items():
            _match = re.findall(r'pns_sensor_adaptor\.nets\.([0-9]+)\.weight', parameter_name)
            if _match:
                _robot_id = _match[0]
                print(f"Sensor channel for the policy: {module.shape[0]}")
                robot_ids_in_file.append(int(_robot_id))
                assert args.cnspns_sensor_channel == module.shape[0], f"Loading from a model with a different number of sensor channels. Want {args.cnspns_sensor_channel}, the model has {module.shape[0]}."
            _match = re.findall(r'pns_motor_adaptor\.nets\.([0-9])+\.weight', parameter_name)
            if _match:
                print(f"Motor channel for the policy: {module.shape[1]}")
                assert args.cnspns_motor_channel == module.shape[1], f"Loading from a model with a different number of motor channels. Want {args.cnspns_motor_channel}, the model has {module.shape[1]}."
        fake_env = DummyVecEnv([gym_interface.make_env(robot_body=_robot_id, wrappers=default_wrapper,
                                            render=False, dataset_folder=args.body_folder) for _robot_id in robot_ids_in_file])
    else:
        fake_env = None
    load_model = model_cls.load(model_filename, fake_env)
    if args.cnspns:
        for robot_id in robot_ids_in_file:
            if robot_id not in bodies:
                model.policy.add_net_to_adaptors(robot_id)
        for robot_id in bodies:
            if robot_id not in robot_ids_in_file:
                load_model.policy.add_net_to_adaptors(robot_id)
    load_weights = load_model.policy.state_dict()
    model.policy.load_state_dict(load_weights)
    # model.policy.rebuild()?
    print(f"Weights loaded from {model_filename}")

    return model

def inspect_regular_parameters(module, subtitle):
    print("="*5, subtitle, "="*5)
    for p in module.parameters():
        if not hasattr(p, "robot_id"):
            p = p.detach().cpu().numpy()
            print(f"shape: {p.shape}")
            if len(p.shape)==1:
                print(p[:3])
            else:
                print(p[:2,:2].flatten())
            
class Log:
    def __init__(self, path="") -> None:
        self.msgs = []
        self.log_path = path
    def record(self, msg):
        self.msgs.append(msg)
    def dump(self):
        if self.log_path=="":
            print("\n".join(self.msgs))
        else:
            with open(self.log_path, "a") as file:
                file.writelines(self.msgs)
        self.msgs = []
