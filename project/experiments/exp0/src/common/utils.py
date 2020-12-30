from genericpath import exists
import os
import pathlib

import numpy as np


def get_exp_folder():
    """Return current experiment folder, such as exp0."""
    _full_path = str(pathlib.Path().absolute())
    _paths = _full_path.split('/')
    assert _paths[-1] == "src", "Project structure has been changed. utils.get_exp_folder() should be changed accordingly."
    assert len(_paths) > 2, "Why path is so short?"
    folder = _paths[-2]

    # Create output folder is not exist yet.
    _path = pathlib.Path("../../../output_data")
    if not _path.is_dir():
        print("Starting a new project? Congratulations! \n\nCreating output data path for the first time.")
        print(f"mkdir {_path.resolve()}")
        _path.mkdir()
    _path = _path / folder
    if not _path.is_dir():
        _path.mkdir(exist_ok=True)
        _subs = ["tensorboard", "plots", "models", "saved_images", "videos", "tmp"]
        for _sub in _subs:
            (_path / _sub).mkdir(exist_ok=True)
    return folder


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
