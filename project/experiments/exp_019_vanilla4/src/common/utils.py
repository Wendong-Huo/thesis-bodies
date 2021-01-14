import os
import hashlib
import pathlib
import yaml
import torch.nn as nn
import numpy as np


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
    _subs = ["tensorboard", "plots", "models", "saved_images", "videos", "checkpoints", "tmp", "bodies"]
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
    if args.vec_normalize:
        filename += "-vnorm"
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
    with (get_input_data_folder() / "hyperparameters.yml").open() as f:
        hp = yaml.load(f, Loader=yaml.SafeLoader)
    hyperparams = hp[conf_name]
    hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    return hyperparams