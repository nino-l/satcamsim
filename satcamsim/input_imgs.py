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
from glob import glob


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


class Comp_processor:
    """Facilitate and simplify operations involving raster data used as comparison."""

    def __init__(self):
        self.open_files = dict()
        return

    def from_config(config, folder_out):
        """
        Initialize Comp_processor with config parameters.

        Parameters
        ----------
        config : Config
            Config specifying the processor parameters.
        folder_out : str
            Path to output folder for the processor.

        Returns
        -------
        processor : Comp_processor
            The initialized processor.

        """
        processor = Comp_processor()
        processor.folder_in = config['COMPARE_FOLDER']
        processor.filenames = config['COMP_FILES']

        # create coordinate reference system
        processor.crs_obj = config['CRS_IN']

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
        self : Comp_processor
            The Comp_processor instance created for the context manager.

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
            except rasterio.errors.RasterioIOError:     # raised if required file not available or not a valid raster file
                print("\r" + "Warning: File " + self.folder_in + filename + " not found or could not be opened by rasterio!", end="")
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
        
        nodata = file.nodatavals[0]
        if nodata is None:
            nodata = 0

        # determine contained pixels
        contained = features.rasterize([(region_to_cut, 1)], out_shape=data.shape[1:], fill=0, transform=window_trans)
        contained_data = np.where(contained, data, nodata)
        return contained_data, window_trans

    def cut_geom(self, geom):
        """
        Cut and save region of passed-in geometry object from raster data.
        The specified geometry is cropped to the edges of the raster data, if necessary.
        Output is saved to the Comp_processor's folder_out.

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
                obj2comp = pyproj.transformer.Transformer.from_crs(self.crs_obj, crs_comp)
                
                # transform geom to comparison data CRS
                vert_coords_transformed = list(obj2comp.itransform(geom.exterior.coords))
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
        if isinstance(geom, Polygon):
            return geom
        
        for _ in range(self.max_iter_glue):
            geom = geom.buffer(self.glue_dist, resolution=4, cap_style=3, join_style=3)     # dilate by self.glue_dist
            geom = unary_union([geom])      # try unifiying all members

            if isinstance(geom, Polygon):
                return geom

        raise ValueError("Dilation limit reached, geometry could not be unified into a single Polygon")


class Input_processor:
    """
    Facilitate and simplify operations involving the input images.

    Parameters
    ----------
    bands : tuple of int
        the band indices to be read from the input data.

    Returns
    -------
    Input_processor : Input_processor
        a new Input_processor instance.

    """

    def __init__(self):
        self.open_files = dict()
        self.available_files = np.array([], dtype=object)
        self.bounds = np.empty((0, 4), dtype=float)
        return

    def from_config(config):
        """
        Initialize a new Input_processor with the parameters specified in config.

        Parameters
        ----------
        config : Config
            Config containing all relevant parameters.

        Returns
        -------
        processor : Input_processor
            The initialized Input_processor instance.

        """
        processor = Input_processor()
        processor.max_open_files = config['MAX_OPEN_FILES']
        processor.folder_in = config['FOLDER_IN']
        processor.file_extension = config['FILEEXTENSION_IN']
        processor.nodata = config['NODATA_IN']
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
        self : Input_processor
            The Input_processor instance for the context manager.

        """
        self.available_files = np.array(glob(self.folder_in + '*' + self.file_extension))
        
        self.bounds = np.full((len(self.available_files), 4), np.nan, dtype=float)
        for idx, file in enumerate(self.available_files):
            with rasterio.open(file) as f:
                self.bounds[idx, :] = f.bounds
                
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
            True if sampling was successful for all requested bands, False otherwise.
        sample : np.array or None
            array of shape (n_bands,) containing mean values for each band if sampling was successful, None otherwise.
        poly : shapely.geometry.Polygon
            the polygon area that was sampled (=ground track coverage). Returned even if sampling was unsuccessful.

        """
        # create polygon of area of pixel projected onto terrain
        poly = Polygon(corners_obj)

        is_success, data_list, trans_list = self.read_data(poly, corners_obj, bands)

        if not is_success:
            return False, None, poly
        
        sum_samples = np.zeros((len(bands),), dtype=float)
        num_samples = 0.
        for data, data_trans in zip(data_list, trans_list):
            # determine contained input pixels in read-in data
            contained = features.rasterize([(poly, 1)], out_shape=data.shape[1:], fill=0, transform=data_trans)
            
            # check nodata
            data = data[:, contained.astype(bool)]
            if self.nodata in data:
                return False, None, poly
    
            sum_samples += data.sum(axis=1)        # sum sampled pixel values per band
            num_samples += contained.sum()         # no. of sampled pixels

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
            contains unique names of the files, ordered by ascending X, if equal by ascending Y positions.

        """
        if not hasattr(objcoords[0], "__len__"):    # if a single point is specified instead of list of points
            return self.get_filenames([objcoords])

        filenames = list()
        
        objcoords = sorted(objcoords)
        for X, Y in objcoords:
            fname = self.available_files[(self.bounds[:, 0] <= X) &  (self.bounds[:, 1] <= Y) & (X <= self.bounds[:, 2]) & (Y <= self.bounds[:, 3])]
            if len(fname) != 1:
                # if file could not be uniquely identified
                return False, None
            
            filenames.append(fname[0])
            
        filenames = list(dict.fromkeys(filenames))  # keep only unique filenames while preserving order
        return True, filenames

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
            if len(self.open_files) >= self.max_open_files:
                self._close_files(len(self.open_files) - self.max_open_files)

            self.open_files[filename] = rasterio.open(filename)
        return

    def read_data(self, poly, vertices, bands):
        """
        Read a rectangular bounding box containing the specified Polygon from input files.
        The current implementation may break if input files use a CRS/transform where not X==East==right, Y==North==up.

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
            Affine transform from raster data pixels to object coordinates, if successful. None otherwise.

        """
        polybounds = poly.bounds
        
        # determine required input files from bounding box corners and try to load them if necessary
        all_files_found, filenames = self.get_filenames([polybounds[0:2],       # SW corner    (yes
                                                         polybounds[0::3],      # NW corner     this
                                                         polybounds[2:0:-1],    # SE corner     is
                                                         polybounds[2:]])       # NE corner     ugly)
        
        if not all_files_found:
            # if coordinates could not be assigned to a unique input file
            return False, None, None
        
        for filename in filenames:
            if filename not in self.open_files:
                self._open_file(filename)
        
        if len(filenames) == 1:
            file = self.open_files[filenames[0]]
            # windowed reading from input file
            poly_window = windows.from_bounds(*polybounds, transform=file.transform)
            data_trans = windows.transform(poly_window, file.transform)
            data = file.read(bands, window=poly_window)
            
            if 0 in data.shape:
                # can happen if only a very thin (< 0.5 input pixels) part of the poly intersects with the input images
                return False, None, None
            return True, [data], [data_trans]

        data_list = []
        trans_list = []
        for filename in filenames:
            file = self.open_files[filename]
            
            # trim bounding box to not extend beyond limits of the file
            # (this will absolutely break if file uses a weird CRS/transform where not X==East==right, Y==North==up:
            # rasterio orders file.bounds as 'left, bottom, right, top', which is not necessarily the same as 'X_min, Y_min, X_max, Y_max'.
            # thats's what geodesysts get for inventing stupid coordinate systems with insane axis ordering, and frankly I won't clean their mess for them)
            windowbounds = [max(polymin, filemin) for polymin, filemin in zip(polybounds[:2], file.bounds[:2])] + [min(polymax, filemax) for polymax, filemax in zip(polybounds[2:], file.bounds[2:])]

            # windowed reading from input file
            poly_window = windows.from_bounds(*windowbounds, transform=file.transform)
            data_trans = windows.transform(poly_window, file.transform)
            data = file.read(bands, window=poly_window)

            if 0 in data.shape:
                # can happen if only a very thin (< 0.5 input pixels) part of the poly intersects with the input images
                continue
            
            data_list.append(data)
            trans_list.append(data_trans)

        return True, data_list, trans_list

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
