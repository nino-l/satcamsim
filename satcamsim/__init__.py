"""`satcamsim` provides a simulator for the images produced by a low Earth orbit remote sensing nanosatellite.
The most important classes and functions are found in the `camera` submodule. Default parameters can be set in `configs`."""

from .camera import Camera, Sensor, Interior_orientation, Cam_pose

import satcamsim.configs as configs

from .input_imgs import Feature_finder, Comp_processor, Input_processor

from .orbit import get_pose_list, filter_poses

from .save_output import Temp_handler, Output_saver

from .support import get_config

import satcamsim.mtf as mtf
