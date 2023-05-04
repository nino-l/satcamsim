"""Provides default config parameters for different parts of the simulator. To change the defaults, edit this file directly."""
from .sim_modes import Sim_modes
import pyproj

NUM_THREADS = 1         #: number of threads, if multithreading (Sim_modes.THREADS) is used
NUM_PROCESSES = 12       #: number of processes, if multiprocessing (Sim_modes.PROCESSES) is used
CHUNK_SIZE = 2**16 * 3  #: size of partial images stored in temp files in pixels, if Sim_modes.CHUNKS is used

#: tuple of `sim_modes.Sim_modes`. Control order of and toggle multiprocessing, multithreading, creation of temp files. Must ALWAYS end with Sim_modes.DEFAULT.
#: Modes are evaluated in order and recursively, i.e. `(Sim_modes.PROCESSES, Sim_modes.THREADS, Sim_modes.DEFAULT)` will spawn `NUM_PROCESSES` sub-processes, which will start `NUM_THREADS` threads EACH (for a total of `NUM_PROCESSES * NUM_THREADS` threads).
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

FOLDER_IN = 'path/to/input/folder/'  #: path to folder containing input images.
CRS_IN = pyproj.CRS('epsg:32632')   #: CRS of input image files and GCPs. Must be a pyproj CRS object. Also referred to as object coordinates
FILEEXTENSION_IN = '.tif'           #: file extension of input images. Files with other extensions in the input image folder will be ignored.
NODATA_IN = 255                       #: nodata value of input images
MAX_OPEN_FILES = 23                 #: max number of input files allowed to be opened at the same time

CRS_ORBIT = pyproj.CRS("epsg:4936") #: CRS of orbit specification. Must be a pyproj CRS object. Only ECEF coordinates (EPSG:4936) are reliably supported by the current implementation of `camera.Cam_pose`

NODATA_OUT = 0          #: nodata value used in output image (must be a valid value for `DTYPE_OUT`)
DTYPE_OUT = 'uint8'    #: dtype of output image, must be a numpy-compatible type (or identifier string) and be compatible with input image values (to prevent overflow etc.)
NOISY = True            #: toggle noise in degraded image

FIND_FEATURES = True    #: toggle localization of GCPs
FEATURE_CSV_PATH_2D = 'path/to/csv/with/2D/GCPs.csv'  #: None or path to .csv with 2D GCPs (containing only X/Y coordinates). File format specified in `input_imgs.Feature_finder`.
FEATURE_CSV_PATH_3D = 'path/to/csv/with/3D/GCPs.csv'  #: None or path to .csv with 3D GCPs (containing X/Y/Z coordinates). File format specified in `input_imgs.Feature_finder`.

COMPARE_FOLDER = 'path/to/raster/data/for/comparisons/'  #: path to folder containing raster data for comparison.
COMP_FILES = None         #: list of filenames for comparison data, or None to use all files in the `COMPARE_FOLDER`
GLUE_MULTIPOLYS = True      #: try to close small gaps in the ground coverage by dilating it up to `MAX_DIST_GLUE`
MAX_ITER_GLUE = 30          #: defines step size in closing gaps as `MAX_DIST_GLUE / MAX_ITER_GLUE`. Large values mean less unnecessary dilation, low values mean faster processing.
MAX_DIST_GLUE = .5          #: maximum dilation of ground track coverage in meters

