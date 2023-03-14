"""Provides default config parameters for different parts of the simulator. To change the defaults, edit this file directly."""
from .sim_modes import Sim_modes

NUM_THREADS = 1         #: number of threads, if multithreading (Sim_modes.THREADS) is used
NUM_PROCESSES = 12       #: number of processes, if multiprocessing (Sim_modes.PROCESSES) is used
CHUNK_SIZE = 2**16 * 3  #: size of partial images stored in temp files in pixels, if Sim_modes.CHUNKS is used

#: tuple of `sim_modes.Sim_modes`. Control order of and toggle multiprocessing, multithreading, creation of temp files. must end with Sim_modes.DEFAULT
SIM_MODES = (Sim_modes.PROCESSES,
             Sim_modes.DEFAULT,)

#: path to a folder where outputs should be saved.
OUTPUT_FOLDER = 'path/to/output/folder/'

#: path to a folder where temp files should be saved.
TEMP_FOLDER = OUTPUT_FOLDER

MEAN_TERRAIN_HEIGHT = 0     #: Z_terrain for simulating images and finding GCPs
APPROX_VIEW_ANGLE = 2.22    #: approximate angle of view of the camera. used only for GCP localization

#: maximum expected off-nadir roll angle (in degrees) over the entire swath.
#: Lower angle speeds up 3D GCP localization, but if higher roll angles occur, GCPs may be missed.
MAX_ROLL_ANGLE = 1


def dop40_filename(X_B, Y_B):
    """
    Generate filenames for DOP with 40cm GSD.

    Parameters
    ----------
    X_B : int
        X coordinate of SE tile corner.
    Y_B : int
        Y coordinate of SE tile corner.

    Returns
    -------
    filename : str
        Filename of the DOP tile with SE corner (X_B, Y_B).

    """
    return f'32{int(X_B/1000)}_{int(Y_B/1000)}.tif'


DOP_FOLDER = 'path/to/DOP/folder/'  #: path to folder containing DOP images.
FILENAME_FORMAT = dop40_filename  #: callable returning a valid filename string for arguments X_B, Y_B. Set to `dop20_filename` for Zugspitze, `dop40_filename` for Ansbach
GSD_DOP = 0.4                     #: GSD of used DOP file

COUNT_R = 2500          #: height of DOP tile in pixels
COUNT_C = 2500          #: width of DOP tile in pixels
MAX_OPEN_FILES = 23     #: max number of DOP files open at the same time

NOISY = True            #: toggle noise in degraded image

FIND_FEATURES = True    #: toggle localization of GCPs
FEATURE_CSV_PATH_2D = 'path/to/csv/with/2D/GCPs.csv'  #: path to .csv with 2D GCPs. File format specified in `input_imgs.Feature_finder`.
FEATURE_CSV_PATH_3D = 'path/to/csv/with/3D/GCPs.csv'  #: path to .csv with 3D GCPs. File format specified in `input_imgs.Feature_finder`.


COMPARE_FOLDER = 'path/to/raster/data/for/comparisons/'  #: path to folder containing raster data for comparison.
GSD_COMP = 3              #: ground sampling distance of comparison data, in meters
COMP_FILES = None         #: list of filenames for comparison data, or None to use all files in the `COMPARE_FOLDER`
GLUE_MULTIPOLYS = True      #: try to close small gaps in the ground coverage by dilating it up to MAX_DIST_GLUE
MAX_ITER_GLUE = 30
MAX_DIST_GLUE = .5          #: maximum dilation of ground track coverage in meters
