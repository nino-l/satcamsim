"""Provides processor classes for reading and processing input data, and localizing GCPs."""

import numpy as np
import rasterio
from rasterio import features, windows
from itertools import product
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from os import listdir
import pyproj
import csv


class Feature_finder:
    """Find position of point features in simulated line scan image.

    The .csv containing 2D GCPs must have the format: `id,X,Y`.

    The .csv containing 3D GCPs must have the format: `id,X,Y,Z`."""

    def __init__(self):
        self.names_2D = np.array([], dtype=object)
        self.points_2D = np.array([], dtype=object)
        self.coords_2D = np.empty((0, 2), dtype=float)

        self.names_3D = np.array([], dtype=object)
        self.points_3D = np.array([], dtype=object)
        self.coords_3D = np.empty((0, 3), dtype=float)

        self.Z_terrain = 0
        self.buffer = 0
        return

    def from_config(config):
        """
        Initialize Feature_finder with config parameters.

        Parameters
        ----------
        config : Config
            Config instance containing desired parameters.

        Returns
        -------
        finder : Feature_finder
            initizialized Feature_finder instance.

        """
        if not config['FIND_FEATURES']:
            return None

        finder = Feature_finder()
        finder.Z_terrain = config['MEAN_TERRAIN_HEIGHT']

        if config['FEATURE_CSV_PATH_2D']:
            filepath = config['FEATURE_CSV_PATH_2D']

            name_list = []
            coord_list = []
            point_list = []
            with open(filepath) as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    name_list.append(str(row[0]))
                    coords = (float(row[1]), float(row[2]))
                    point_list.append(Point(coords))
                    coord_list.append(coords)

            coord_list, point_list, name_list = zip(*sorted(zip(coord_list, point_list, name_list)))

            finder.coords_2D = np.array(coord_list)
            finder.names_2D = np.empty(len(name_list), dtype=object)
            finder.points_2D = np.empty(len(point_list), dtype=type(Point))
            for idx, (point, name) in enumerate(zip(point_list, name_list)):
                finder.points_2D[idx] = point
                finder.names_2D[idx] = name

        if config['FEATURE_CSV_PATH_3D']:
            filepath = config['FEATURE_CSV_PATH_3D']

            # read in 3D GCPs
            name_list = []
            coord_list = []
            point_list = []
            with open(filepath) as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    name_list.append(str(row[0]))
                    coords = (float(row[1]), float(row[2]), float(row[3]))
                    point_list.append(Point(coords))
                    coord_list.append(coords)

            coord_list, point_list, name_list = zip(*sorted(zip(coord_list, point_list, name_list)))

            finder.coords_3D = np.array(coord_list)     # fill array of point coordinates
            finder.names_3D = np.empty(len(name_list), dtype=object)
            finder.points_3D = np.empty(len(point_list), dtype=type(Point))
            for idx, (point, name) in enumerate(zip(point_list, name_list)):
                finder.points_3D[idx] = point           # fill array of Point objects
                finder.names_3D[idx] = name             # fill array of point names/identifiers as str

            approx_view_angle = np.deg2rad(config['APPROX_VIEW_ANGLE'])
            max_roll = np.deg2rad(config['MAX_ROLL_ANGLE'])

            # buffer to determine radius within which distorted 3D points may be found
            finder.buffer = np.max(np.abs(finder.coords_3D[:, 2] - finder.Z_terrain)) * np.tan(0.5 * approx_view_angle + max_roll)
        return finder

    def check(self, poly, vertices, line_idx, px_idx, sensor_name, pose):
        """
        Check which of the Feature_finder's GCPs (if any) are contained within a polygon.

        Parameters
        ----------
        poly : Polygon
            Polygon to be checked.
        line_idx : int
            current pose.idx.
        px_idx : int
            current pixel's c coordinate.
        sensor_name : str
            current sensor's name.
        pose : Cam_pose
            current sensor's exterior orientation.

        Returns
        -------
        found_feats : list[tuple]
            list containing tuples with (GCP_name, line_idx, px_idx, sensor_name, GCP_X, GCP_Y, GCP_Z) of all GCPs contained in poly.
            GCPs without Z information are marked with GCP_Z = 999999.

        """
        X_min, Y_min, X_max, Y_max = poly.bounds

        candidate_idxs = self._get_candidates_2D(X_min, Y_min, X_max, Y_max)
        # create list with tuple of format: (feature_name, row, column, sensor_name, feature_X, feature_Y, 999999)
        found_feats_2D = [(point_name, line_idx, px_idx, sensor_name, point_coords[0], point_coords[1], 999999) for point_name, point_coords, point in zip(self.names_2D[candidate_idxs], self.coords_2D[candidate_idxs], self.points_2D[candidate_idxs]) if poly.contains(point)]

        candidate_idxs = self._get_candidates_3D(X_min, Y_min, X_max, Y_max)
        # create list with tuple of format: (feature_name, row, column, sensor_name, feature_X, feature_Y, feature_Z)
        found_feats_3D = [(point_name, line_idx, px_idx, sensor_name, point_coords[0], point_coords[1], point_coords[2]) for point_name, point_coords, point in zip(self.names_3D[candidate_idxs], self.coords_3D[candidate_idxs], self.points_3D[candidate_idxs]) if self._get_poly_Z(vertices, point_coords[2], pose).contains(point)]

        return found_feats_2D + found_feats_3D

    def _get_candidates_3D(self, X_min, Y_min, X_max, Y_max):
        """Preselect GCPs based on rectangular bounding box and buffer zone."""
        start_idx = np.searchsorted(self.coords_3D[:, 0], X_min - self.buffer)
        stop_idx = start_idx + np.searchsorted(self.coords_3D[start_idx:, 0], X_max + self.buffer, 'right')

        candidate_idxs = start_idx + np.where(np.logical_and(self.coords_3D[start_idx:stop_idx, 1] >= Y_min - self.buffer, self.coords_3D[start_idx:stop_idx, 1] <= Y_max + self.buffer))[0]

        return candidate_idxs

    def _get_candidates_2D(self, X_min, Y_min, X_max, Y_max):
        """Preselect GCPs based on rectangular bounding box."""
        start_idx = np.searchsorted(self.coords_2D[:, 0], X_min)
        stop_idx = start_idx + np.searchsorted(self.coords_2D[start_idx:, 0], X_max, 'right')

        candidate_idxs = start_idx + np.where(np.logical_and(self.coords_2D[start_idx:stop_idx, 1] >= Y_min, self.coords_2D[start_idx:stop_idx, 1] <= Y_max))[0]

        return candidate_idxs

    def _get_ray_at_Z(self, point, proj_center, Z):
        """Ray tracing for 3D GCP detection. Returns X, Y position of ray passing through point and proj_center at elevation Z."""
        deltaZ = proj_center[2] - self.Z_terrain
        deltas = (proj_center[0:2] - point) / deltaZ
        X, Y = point + deltas * (Z - self.Z_terrain)
        return X, Y

    def _get_poly_Z(self, vertices, Z, pose):
        """Create new Polygon at a candidate GCP's Z elevation to determine whether it is inside FOV."""
        new_center = pose.XYZ_0
        old_center = pose.previous.XYZ_0
        poly_Z = Polygon([self._get_ray_at_Z(vertices[0], new_center, Z),
                          self._get_ray_at_Z(vertices[1], old_center, Z),
                          self._get_ray_at_Z(vertices[2], old_center, Z),
                          self._get_ray_at_Z(vertices[3], new_center, Z)])
        return poly_Z


class Raster_processor:
    """Facilitate and simplify operations involving raster data used as comparison."""

    def __init__(self):
        self.open_files = dict()
        return

    def from_config(config, folder_out):
        """
        Initialize Raster_processor with config parameters.

        Parameters
        ----------
        config : Config
            Config specifying the processor parameters.
        folder_out : str
            Path to output folder for the processor.

        Returns
        -------
        processor : Raster_processor
            The initialized processor.

        """
        processor = Raster_processor()
        processor.gsd_in = config['GSD_COMP']
        processor.folder_in = config['COMPARE_FOLDER']
        processor.filenames = config['COMP_FILES']

        # create coordinate reference system
        processor.crs_DOP = pyproj.CRS("epsg:25832")

        processor.folder_out = folder_out

        processor.glue_multipolys = config['GLUE_MULTIPOLYS']
        if processor.glue_multipolys:
            processor.max_dist_glue = config['MAX_DIST_GLUE']
            processor.max_iter_glue = config['MAX_ITER_GLUE']
            processor.glue_dist = processor.max_dist_glue / processor.max_iter_glue

        return processor

    def __del__(self):
        """
        Guarantee that all files are closed before deletion.

        Returns
        -------
        None.

        """
        for _, file in self.open_files.items():
            file.close()
        return

    def __enter__(self):
        """
        Open comparison files for use in context manager.

        Returns
        -------
        self : Raster_processor
            The Raster_processor instance created for the context manager.

        """
        if not self.filenames:
            self.filenames = listdir(self.folder_in)
        for filename in self.filenames:
            self._open_file(filename)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensure proper closing of all files when exiting context manager.

        Parameters
        ----------
        exc_type : exception type or None
            Type of exception raised within context manager, or None.
        exc_value : Exception or None
            The exception raised within context manager, or None.
        traceback : Traceback or None
            Traceback object for the raised exception, or None.

        Returns
        -------
        None.

        """
        for _, file in self.open_files.items():
            file.close()
        return

    def _open_file(self, filename):
        """
        Open specified files from disk.

        Parameters
        ----------
        filename : str
            name of file to be loaded.

        Returns
        -------
        flag : bool
            True if file was successfully opened, False otherwise.

        """
        if filename not in self.open_files:
            try:
                self.open_files[filename] = rasterio.open(self.folder_in + filename)
            except rasterio.errors.RasterioIOError:     # raised if required file not available
                print("\r" + "Warning: File " + filename + " not found!", end="")
                return False
        return True

    def read_from_file(self, region_to_cut, file):
        """
        Reads data contained in the specified region from file.

        Parameters
        ----------
        region_to_cut : Polygon
            Region of interest, in XY object coordinates.
        file : rasterio.DatasetReader
            file from which data is to be read.

        Returns
        -------
        contained_data : np.ndarray
            extracted raster data.
        window_trans : rasterio.affine.Affine
            transform of the extracted raster data section.

        """
        X_min, Y_min, X_max, Y_max = region_to_cut.bounds
        poly_window = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file.transform)
        window_trans = windows.transform(poly_window, file.transform)
        data = file.read(window=poly_window)

        # determine contained pixels
        contained = features.rasterize([(region_to_cut, 1)], out_shape=data.shape[1:], fill=0, transform=window_trans)
        contained_data = np.where(contained, data, 0)
        return contained_data, window_trans

    def cut_geom(self, geom):
        """
        Cut and save region of passed-in geometry object from raster data.
        The specified geometry is cropped to the edges of the raster data, if necessary.
        Output is saved to the Raster_processor's folder_out.

        Parameters
        ----------
        geom : shapely.geometry object
            geometry object describing the region to be extracted, in EPSG:25832 coordinates.

        Returns
        -------
        None.

        """
        with rasterio.Env() as rio_env:
            if self.glue_multipolys:
                geom = self.glue_geoms(geom)    # close small gaps in the coverage

            for filename, file in self.open_files.items():
                
                # CRS transformation
                crs_comp = pyproj.CRS.from_user_input(file.crs)
                DOP2comp = pyproj.transformer.Transformer.from_crs(self.crs_DOP, crs_comp)
                
                # transform geom to comparison data CRS
                vert_coords_transformed = list(DOP2comp.itransform(geom.exterior.coords))
                vertices_transformed = [Point(X, Y) for X, Y in vert_coords_transformed]
                geom_transformed = Polygon(vertices_transformed)
                
                # crop region to edges of comparison data
                X_lo, Y_lo, X_hi, Y_hi = file.bounds
                region_to_cut = geom_transformed.intersection(Polygon([(X_lo, Y_lo), (X_hi, Y_lo), (X_hi, Y_hi), (X_lo, Y_hi)]))

                # if no overlap exists
                if not region_to_cut:
                    continue

                contained_data, window_trans = self.read_from_file(region_to_cut, file)
                # write to new raster file
                profile = file.profile
                profile.update(height=contained_data.shape[1],
                               width=contained_data.shape[2],
                               transform=window_trans)
                with rasterio.open(self.folder_out + 'cutout_' + filename, 'w', **profile) as dest:
                    for idx, band_data in enumerate(contained_data, start=1):
                        dest.write(band_data, idx)
        return

    def glue_geoms(self, geom):
        """
        Dilate a collection of geometries iteratively until its members can be merged into a single Polygon.
        If a single, non-Polygon geometry object is passed, it is dilated and therefore converted to a Polygon.
        If not all members can be unified into a single Polygon within the dilation limits, an exception is raised.

        Parameters
        ----------
        geom : shapely.geometry object
            geometry collection or single geometry object.

        Returns
        -------
        geom : shapely.geometry object
            single Polygon, if glueing was successful.

        """
        for _ in range(self.max_iter_glue):
            if isinstance(geom, Polygon):
                return geom

            geom = geom.buffer(self.glue_dist, resolution=4, cap_style=3, join_style=3)     # dilate by self.glue_dist
            geom = unary_union([geom])      # try unifiying all members

        if not isinstance(geom, Polygon):
            raise ValueError("Dilation limit reached, geometry could not be unified into a single Polygon")


class DOP_processor:
    """
    Facilitate and simplify operations involving the input DOPs.

    Parameters
    ----------
    bands : tuple of int
        the band indices to be read from the input data.

    Returns
    -------
    dop_processor : DOP_processor
        a new DOP_processor instance.

    """

    def __init__(self):
        self.open_files = dict()
        self.available_files = list()
        return

    def from_config(config):
        """
        Initialize a new DOP_processor with the parameters specified in config.

        Parameters
        ----------
        config : Config
            Config containing all relevant parameters.
        bands : tuple of int
            The relevant band indices of the input data.

        Returns
        -------
        processor : DOP_processor
            The initialized DOP_processor instance.

        """
        processor = DOP_processor()
        processor.max_open_files = config['MAX_OPEN_FILES']
        processor.folder_in = config['DOP_FOLDER']
        processor.gsd_in = config['GSD_DOP']
        processor.count_R = config['COUNT_R']
        processor.count_C = config['COUNT_C']
        processor.filename_format = config['FILENAME_FORMAT']

        processor.DOP_width = processor.count_C * processor.gsd_in
        processor.DOP_height = processor.count_R * processor.gsd_in

        return processor

    def __del__(self):
        """
        Guarantee that all files are closed before deletion.

        Returns
        -------
        None.

        """
        for _, file in self.open_files.items():
            file.close()
        return

    def __enter__(self):
        """
        Return self for use in context manager.

        Returns
        -------
        self : DOP_processor
            The DOP_processor instance for the context manager.

        """
        self.available_files = listdir(self.folder_in)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensure proper closing of all files when exiting context manager.

        Parameters
        ----------
        exc_type : exception type or None
            Type of exception raised within context manager, or None.
        exc_value : Exception or None
            The exception raised within context manager, or None.
        traceback : Traceback or None
            Traceback object for the raised exception, or None.

        Returns
        -------
        None.

        """
        for _, file in self.open_files.items():
            file.close()
        return

    def sample_area(self, corners_obj, bands):
        """
        Compute mean of raster data within the polygonal area specified by its corners in object coordinates.

        Parameters
        ----------
        corners_obj: list of tuple
            list of (X, Y) coordinate tuples representing polygon corners.
        bands : tuple of int
            tuple of the band indices to be read.

        Returns
        -------
        flag : bool
            True if sampling was successful, False otherwise.
        sample : np.array or None
            array of shape (n_bands,) containing mean values for each band if sampling was successful, None otherwise.
        poly : shapely.geometry.Polygon
            the polygon area that was sampled (=ground track coverage). Returned even if sampling was unsuccessful.

        """
        # create polygon of area of pixel projected onto terrain
        poly = Polygon(corners_obj)

        is_success, data, data_trans = self.read_data(poly, corners_obj, bands)

        if not is_success:
            return False, None, poly

        # determine contained input pixels in read-in data
        contained = features.rasterize([(poly, 1)], out_shape=data.shape[1:], fill=0, transform=data_trans)

        # R_contained, C_contained = np.nonzero(contained)
        sum_samples = np.where(contained, data, 0).sum(axis=(1, 2))    # sum sampled pixel values
        num_samples = np.sum(contained)                                # add no. of sampled pixels

        sample = sum_samples / num_samples        # compute mean sampled pixel values

        return True, sample, poly

    def get_filenames(self, objcoords):
        """
        Construct all file names containing specified points.

        Parameters
        ----------
        objcoords : list[tuple]
            list of point coordinate tuples in (X, Y) format.

        Returns
        -------
        filenames : list[str]
            contains names of the files, ordered by ascending X, if equal by ascending Y positions.

        """
        if not hasattr(objcoords[0], "__len__"):    # if a single point is specified instead of list of points
            return self.get_filenames([objcoords]).pop()

        filenames = list()

        X, Y = zip(*objcoords)
        X_min = min(X)
        Y_min = min(Y)
        X_max = max(X)
        Y_max = max(Y)

        # southwest corner coordinates coordinates of most southwest DOP tile
        X_B0 = int(X_min / self.DOP_width) * self.DOP_width
        Y_B0 = int(Y_min / self.DOP_width) * self.DOP_width

        # base coordinates for all required DOP tiles
        X_Bs = np.arange(X_B0, X_max, self.DOP_width, dtype=int)
        Y_Bs = np.arange(Y_B0, Y_max, self.DOP_height, dtype=int)
        if not X_Bs.size:
            # if all X values are equal, np.arange will return np.array([])
            X_Bs = [int(X_B0)]
        if not Y_Bs.size:
            # if all Y values are equal, np.arange will return np.array([])
            Y_Bs = [int(Y_B0)]

        base_coords = list(product(X_Bs, Y_Bs))

        for X_B, Y_B in base_coords:
            filename = self.filename_format(X_B, Y_B)
            filenames.append(filename)
        return filenames

    def _open_file(self, filename):
        """
        Open specified files from disk.

        Parameters
        ----------
        filename : str
            name of file to be loaded.

        Returns
        -------
        flag : bool
            True if file was successfully opened, False otherwise.

        """
        if filename not in self.open_files:
            if filename not in self.available_files:
                return False

            if len(self.open_files) >= self.max_open_files:
                self._close_files(len(self.open_files) - self.max_open_files)

            self.open_files[filename] = rasterio.open(self.folder_in + filename)
        return True

    def read_data(self, poly, vertices, bands):
        """
        Read a rectangular bounding box containing the specified Polygon from DOP files.
        Sections of 1, 2, or 4 files are compiled, if necessary.

        Parameters
        ----------
        poly : shapely.geometry.Polygon
            Polygonal ROI which will be fully covered by the read-in data.
        vertices : np.ndarray
            coordinates of ROI vertices.
        bands : tuple of int
             tuple of the band indices to be read.

        Returns
        -------
        flag : bool
            True if data was read in successfully, False otherwise.
        data : np.ndarray or None
            raster data contained in the rectangular bounding box around poly, if successful. None otherwise.
        data_trans : rasterio.affine.Affine or None
            Affine transform from raster data pixels to object coordinates, if successful. None otherwise..

        """
        # determine required input files and try to load them if necessary
        filenames = self.get_filenames(vertices)
        for filename in filenames:
            if filename not in self.open_files:
                is_success = self._open_file(filename)

                if not is_success:              # if file could not be loaded
                    return False, None, None

        X_min, Y_min, X_max, Y_max = poly.bounds

        try:
            if len(filenames) == 1:
                file = self.open_files[filenames[0]]

                # windowed reading from input file
                poly_window = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file.transform)
                data_trans = windows.transform(poly_window, file.transform)
                data = file.read(bands, window=poly_window)

            elif len(filenames) == 2:
                file_SW = self.open_files[filenames[0]]     # southern or western file is first in list
                window_SW = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_SW.transform)

                # create empty output array from window size
                data = np.full((len(bands), round(window_SW.height), round(window_SW.width)), np.NaN, np.uint8)
                data_trans = windows.transform(window_SW, file_SW.transform)    # Affine transform for data array

                data_SW = file_SW.read(bands, window=window_SW)        # read windowed data from file_SW
                _, h_SW, w_SW = data_SW.shape
                if h_SW and w_SW:
                    data[:, -h_SW:, 0:w_SW] = data_SW           # fill in appropriate area of output array

                file_NE = self.open_files[filenames[1]]     # northern or eastern file is next in list
                window_NE = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_NE.transform)
                data_NE = file_NE.read(bands, window=window_NE)        # read windowed data from file_NE
                _, h_NE, w_NE = data_NE.shape
                if h_NE and w_NE:
                    data[:, 0:h_NE, -w_NE:] = data_NE       # fill in appropriate area of output array

            elif len(filenames) == 4:
                file_SW = self.open_files[filenames[0]]
                window_SW = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_SW.transform)

                data = np.full((len(bands), round(window_SW.height), round(window_SW.width)), np.NaN, np.uint8)
                data_trans = windows.transform(window_SW, file_SW.transform)

                data_SW = file_SW.read(bands, window=window_SW)
                _, h_SW, w_SW = data_SW.shape
                if h_SW and w_SW:
                    data[:, -h_SW:, 0:w_SW] = data_SW

                file_SE = self.open_files[filenames[2]]
                window_SE = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_SE.transform)
                data_SE = file_SE.read(bands, window=window_SE)
                _, h_SE, w_SE = data_SE.shape
                if h_SE and w_SE:
                    data[:, -h_SE:, -w_SE:] = data_SE

                file_NW = self.open_files[filenames[1]]
                window_NW = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_NW.transform)
                data_NW = file_NW.read(bands, window=window_NW)
                _, h_NW, w_NW = data_NW.shape
                if h_NW and w_NW:
                    data[:, 0:h_NW, 0:w_NW] = data_NW

                file_NE = self.open_files[filenames[3]]
                window_NE = windows.from_bounds(X_min, Y_min, X_max, Y_max, transform=file_NE.transform)
                data_NE = file_NE.read(bands, window=window_NE)
                _, h_NE, w_NE = data_NE.shape
                if h_NE and w_NE:
                    data[:, 0:h_NE, -w_NE:] = data_NE

            else:
                raise ValueError("poly must be covered by one single, two neighboring, or 2x2 neighboring files")

            return True, data, data_trans

        except KeyError:
            return False, None, None

    def _close_files(self, n_files=-1):
        """
        Close files that have been open the longest (oldest file in dict).

        Parameters
        ----------
        n_files : int, optional
            Number of files to close, or -1 to close all files. The default is -1.
            Note: if n_files >= len(self.open_files), all files will be closed.

        Returns
        -------
        None.

        """
        if n_files == -1 or n_files > len(self.open_files):
            n_files = len(self.open_files)

        for _ in range(n_files):
            filename, file = next(iter(self.open_files.items()))
            file.close()
            del self.open_files[filename]
        return
