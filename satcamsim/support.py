"""Supporting functions for the simulator."""
import numpy as np
import satcamsim.configs as configs


class Config(dict):
    """Special dictionary containing config parameters."""

    def __init__(self, params):
        for key, val in params.items():
            if key.isupper() and not key.startswith('_'):
                self[key] = val
        return

    def __str__(self):
        return '\n'.join([("        {:<25s}" + str(value)).format(param) for param, value in self.items()])


def get_config():
    """
    Creates a `Config` instance based on default parameters set in `configs.py`.

    Returns
    -------
    default_config : Config
        Config instance containing default parameters.

    """
    return Config(configs.__dict__)


def rotation_x(angle):
    R = np.array([[1, 0, 0],
                 [0, np.cos(angle), -np.sin(angle)],
                  [0, np.sin(angle), np.cos(angle)]])
    return R


def rotation_y(angle):
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    return R


def rotation_z(angle):
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    return R
