"""Provides Camera and Sensor classes with different methods for image simulation, as well as auxiliary classes Cam_pose and Interior_orientation."""

import numpy as np
from itertools import chain
from scipy.signal import convolve2d
import pyproj
import rasterio
from shapely.geometry import Polygon
from shapely.ops import unary_union
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from .input_imgs import DOP_processor, Feature_finder
from .save_output import Temp_handler
from .support import get_config, rotation_x, rotation_y, rotation_z
from .sim_modes import Sim_modes


class Camera:
    """
    Camera object with pose and sensors attributes.

    Returns
    -------
    None.

    """

    def __init__(self):
        self.sensors = []
        self.pose = None
        return

    def add_sensor(self, sensor):
        """
        Append additional Sensor object to Camera's list of sensors.

        Parameters
        ----------
        sensor : Sensor
            sensor to be appended.

        Returns
        -------
        self : Camera
            updated camera instance.

        """
        self.sensors.append(sensor)
        return self

    def set_pose(self, pose):
        """
        Specify exterior orientation of camera.

        Parameters
        ----------
        pose : Cam_pose
            pose of the camera

        Returns
        -------
        self : Camera
            updated camera instance.
        """
        self.pose = pose
        for sensor in self.sensors:
            sensor.pose = self.pose
        return self

    def take_line_img(self, dop_processor, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Acquire a single line image from the current pose using the specified sensors.

        Parameters
        ----------
        dop_processor : DOP_processor
            DOP_processor instance used to read data from DOP.
        active_sensors : list of Sensor, optional
            Sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.ndarray of np.uint8
            array of simulated images, shape (n_sensors, n_bands, 1, n_pixels).
        coverage : Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        if active_sensors is None:
            active_sensors = self.sensors

        if out_array is None:
            num_bands = max([len(sensor.bands) for sensor in active_sensors])
            num_pixels = max([sensor.pixels for sensor in active_sensors])
            img_out = np.full((len(active_sensors), num_bands, num_pixels), np.NaN, np.uint8)
        else:
            img_out = out_array

        coverage_geoms = list()
        found_feats = list()

        for sensor_idx, sensor in enumerate(active_sensors):
            # take line images with all specified sensors
            img_out[sensor_idx, :, :], new_coverage, new_feats = sensor.take_line_img(dop_processor, config, feat_finder)

            # add ground covered in new line image
            coverage_geoms.append(new_coverage)
            found_feats += new_feats

        coverage = unary_union(coverage_geoms)  # union of ground track coverage of individual pixels
        return img_out, coverage, found_feats

    def default_swath(self, pose_list, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Take image composed of line scans from specified poses.

        Parameters
        ----------
        pose_list : list of Cam_pose
            List of poses to be used for the individual line images.
        active_sensors : list of Sensor, optional
            sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            array of simulated images, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        with rasterio.Env() as rio_env:
            if active_sensors is None:
                active_sensors = self.sensors

            num_lines = len(pose_list)                                           # height of output image

            if out_array is None:
                num_bands = max([len(sensor.bands) for sensor in active_sensors])    # number of channels of output image
                num_pixels = max([sensor.pixels for sensor in active_sensors])       # width if output image

                # create array containing output images for all sensors and bands
                img_out = np.full((len(active_sensors), num_bands, num_lines, num_pixels), np.NaN, np.uint8)
            else:
                img_out = out_array

            coverage_geoms = list()
            found_feats = list()

            with DOP_processor.from_config(config) as dop_processor:
                for pose_idx, pose in enumerate(pose_list):
                    # update pose and take new line image; write directly to output array
                    _, new_coverage, new_feats = self.set_pose(pose).take_line_img(dop_processor, active_sensors, out_array=img_out[:, :, pose_idx, :], config=config, feat_finder=feat_finder)
                    coverage_geoms.append(new_coverage)
                    found_feats += new_feats

            coverage = unary_union(coverage_geoms)  # union of ground track coverage of individual scan lines
            return img_out, coverage, found_feats

    def chunky_swath(self, pose_list, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Acquire image in chunks of pre-set approximate size, and store each chunk in a temp file to prevent RAM/CPU chache from filling up.

        Parameters
        ----------
        pose_list : list of Cam_pose
            List of poses to be used for the individual line images.
        active_sensors : list of Sensor, optional
            sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            array of simulated images, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        if active_sensors is None:
            active_sensors = self.sensors

        chunk_size = config['CHUNK_SIZE']
        num_lines = len(pose_list)                                           # height of output image

        num_bands = max([len(sensor.bands) for sensor in active_sensors])    # number of channels of output image
        num_pixels = max([sensor.pixels for sensor in active_sensors])       # width of output image

        lines_per_chunk = int(chunk_size / (num_bands * num_pixels))
        num_chunks = int(np.ceil(num_lines / lines_per_chunk))

        # split input parameters into chunks
        splits = [i * lines_per_chunk for i in range(num_chunks)]
        splits.append(num_lines)

        with Temp_handler.from_config(config) as temp_handler:
            # sequentially simulate chunks and store in temp files
            for i in range(num_chunks):
                temp_handler.save(self.take_swath_img_raw(pose_list[splits[i]:splits[i + 1]], active_sensors, None, config, feat_finder))

            # load partial results from temp files
            imgs, coverage_geoms, split_feats = zip(*temp_handler.load_all())

        if out_array is None:
            # create array containing output images for all sensors and bands
            img_out = np.full((len(active_sensors), num_bands, num_lines, num_pixels), np.NaN, np.uint8)
        else:
            img_out = out_array

        # write individual chunks to one output image
        for i in range(num_chunks):
            img_out[:, :, splits[i]:splits[i + 1], :] = imgs[i]

        coverage = unary_union(coverage_geoms)          # union of ground track coverage from individual chunks
        found_feats = chain.from_iterable(split_feats)  # compile detected GCPs into single list
        return img_out, coverage, found_feats

    def multithread_swath(self, pose_list, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Take image composed of line scans from specified poses using multithreading.
        If possible, use multiprocessing instead for better performance.

        Parameters
        ----------
        pose_list : list of Cam_pose
            List of poses to be used for the individual line images.
        active_sensors : list of Sensor, optional
            sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            array of simulated images, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        if active_sensors is None:
            active_sensors = self.sensors

        num_threads = config['NUM_THREADS']
        num_lines = len(pose_list)                                           # height of output image

        if out_array is None:
            num_bands = max([len(sensor.bands) for sensor in active_sensors])    # number of channels of output image
            num_pixels = max([sensor.pixels for sensor in active_sensors])       # width if output image

            # create array containing output images for all sensors and bands
            img_out = np.full((len(active_sensors), num_bands, num_lines, num_pixels), np.NaN, np.uint8)
        else:
            img_out = out_array

        # split input parameters into chunks (one per process)
        split_len = int(num_lines / num_threads)
        splits = [i * split_len for i in range(num_threads)]
        splits.append(num_lines)

        split_args = [(pose_list[splits[i]:splits[i + 1]], active_sensors, img_out[:, :, splits[i]:splits[i + 1], :], config, feat_finder)
                      for i in range(num_threads)]

        # pool of threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            split_results = executor.map(lambda args: self.take_swath_img_raw(*args), split_args)

        _, coverage_geoms, split_feats = zip(*split_results)
        coverage = unary_union(coverage_geoms)          # union of ground track coverage from individual threads
        found_feats = chain.from_iterable(split_feats)  # compile detected GCPs into single list
        return img_out, coverage, found_feats

    def multiprocess_swath(self, pose_list, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Take image composed of line scans from specified poses using multiprocessing.
        Note that on Windows, the script running the simulation must be guarded by if __name__ == '__main__'

        Parameters
        ----------
        pose_list : list of Cam_pose
            List of poses to be used for the individual line images.
        active_sensors : list of Sensor, optional
            sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            array of simulated images, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        if active_sensors is None:
            active_sensors = self.sensors

        num_lines = len(pose_list)                                           # height of output image

        num_bands = max([len(sensor.bands) for sensor in active_sensors])    # number of channels of output image
        num_pixels = max([sensor.pixels for sensor in active_sensors])       # width of output image
        num_processes = config['NUM_PROCESSES']

        # split input parameters into chunks (one per process)
        split_len = int(num_lines / num_processes)
        splits = [i * split_len for i in range(num_processes)]
        splits.append(num_lines)

        split_args = [(pose_list[splits[i]:splits[i + 1]], active_sensors, None, config, feat_finder) for i in range(num_processes)]

        # pool of processes
        with mp.Pool() as pool:
            split_results = pool.starmap(self.take_swath_img_raw, split_args)

        imgs, coverage_geoms, split_feats = zip(*split_results)

        if out_array is None:
            # create array containing output images for all sensors and bands
            img_out = np.full((len(active_sensors), num_bands, num_lines, num_pixels), np.NaN, np.uint8)
        else:
            img_out = out_array

        # write to output image
        for i, img in enumerate(imgs):
            img_out[:, :, splits[i]:splits[i + 1], :] = img

        coverage = unary_union(coverage_geoms)          # union of ground track coverage from individual processes
        found_feats = chain.from_iterable(split_feats)  # compile detected GCPs into single list
        return img_out, coverage, found_feats

    def take_swath_img_raw(self, pose_list, active_sensors=None, out_array=None, config=get_config(), feat_finder=None):
        """
        Take image composed of line scans from specified poses, without any degradation effects.
        Maps arguments to the corresponding Sim_mode.

        Parameters
        ----------
         pose_list : list of Cam_pose
             List of poses to be used for the individual line images.
         active_sensors : list of Sensor, optional
             sensor objects to be used for the image. If not given, all sensors
             are used.
         out_array : np.ndarray, optional
             output array for in-place writing. If not given, a new array is created.
        config : Config, optional
             Config dict containing parameters for the simulation run. If not provided, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            array of simulated images, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.

        """
        config = config.copy()
        sim_modes = config['SIM_MODES']
        sim_mode = sim_modes[0]
        config['SIM_MODES'] = sim_modes[1:]

        if sim_mode == Sim_modes.DEFAULT:
            if sim_modes[1:]:
                raise ValueError('sim_modes must not specify additional simulation modes following a Sim_modes.DEFAULT entry')
            return self.default_swath(pose_list, active_sensors, out_array, config, feat_finder)

        elif sim_mode == Sim_modes.PROCESSES:
            return self.multiprocess_swath(pose_list, active_sensors, out_array, config, feat_finder)

        elif sim_mode == Sim_modes.THREADS:
            return self.multithread_swath(pose_list, active_sensors, out_array, config, feat_finder)

        elif sim_mode == Sim_modes.CHUNKS:
            return self.chunky_swath(pose_list, active_sensors, out_array, config, feat_finder)

        else:
            raise ValueError('sim_modes must only contain members of camera.Sim_modes')

    def simulate_swath(self, pose_list, active_sensors=None, out_array=None, PSF=None, config=get_config()):
        """
        Simulate image composed of line scans from specified poses, and artificially degrade it as specified by PSF and Sensor.SNR.

        Parameters
        ----------
        pose_list : list of Cam_pose
            List of poses to be used for the individual line images.
        active_sensors : list of Sensor, optional
            sensor objects to be used for the image. If not given, all sensors
            are used.
        out_array : np.ndarray, optional
            output array for in-place writing. If not given, a new array is created.
        PSF : np.ndarray, optional
            Filter mask representing the point spread function, sampled at the pixel grid nodes. If not given, no filter is applied.
            If PSF.ndim == 2, all bands of all sensors are degraded with the same PSF.
            If PSF.ndim == 3, all bands of each sensor are degraded with the same PSF, and must have PSF.shape[0] == n_bands.
            If PSF.ndim == 4, every band of every sensor is degraded with an individual PSF, and must have PSF.shape[0] == sensors, PSF.shape[1] == n_bands.
        config : Config, optional
            Config dict containing parameters for the simulation run. If not provided, default parameters are used.

        Returns
        -------
        img_out : np.array
            array of simulated images after degradation according to config, shape (n_sensors, n_bands, n_lines, n_pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
        found_feats : list
            list of found GCPs.
        img_raw : np.array
            array of simulated images without added filter/noise, shape (n_sensors, n_bands, n_lines, n_pixels).

        """
        if config['FIND_FEATURES']:
            feat_finder = Feature_finder.from_config(config)
        else:
            feat_finder = None

        img_raw, coverage, found_feats = self.take_swath_img_raw(pose_list, active_sensors, out_array, config, feat_finder)

        if config['NOISY'] or (PSF is not None):
            SNR = np.array([sensor.SNRs for sensor in self.sensors])
            img_out = Camera.postprocess(img_raw, PSF, SNR, config)
        else:
            img_out = img_raw

        return img_out, coverage, found_feats, img_raw

    def postprocess(img_raw, PSF=None, SNR=None, config=get_config()):
        """
        Artificially degrade (filter and noisify) simulated image.

        Parameters
        ----------
        img_raw : np.array
            array of simulated images without added filter/noise, shape (n_sensors, n_bands, n_lines, n_pixels).
        PSF : np.ndarray, optional
            Filter mask representing the point spread function, sampled at the pixel grid nodes. If not given, no filter is applied.
            If PSF.ndim == 2, all bands of all sensors are degraded with the same PSF.
            If PSF.ndim == 3, all bands of each sensor are degraded with the same PSF, and must have PSF.shape[0] == n_bands.
            If PSF.ndim == 4, every band of every sensor is degraded with an individual PSF, and must have PSF.shape[0] == sensors, PSF.shape[1] == n_bands.
        SNR : np.ndarray, optional
            Signal-to-noise ratios for addition of AWGN. Same conventions for shape as PSF. If not given, no noise is added.
        config : Config, optional
            Config dict containing parameters for the simulation run. If not given, default parameters are used.

        Returns
        -------
        img_out : np.array
            array of simulated images after degradation, shape (n_sensors, n_bands, n_lines, n_pixels).

        """
        img_out = img_raw.copy()

        # apply PSF
        if PSF is not None:
            while PSF.ndim < img_out.ndim:
                PSF = PSF[None, :]

            if PSF.shape[0] != img_out.shape[0]:
                PSF = np.repeat(PSF, img_out.shape[0], axis=0)

            if PSF.shape[1] != img_out.shape[1]:
                PSF = np.repeat(PSF, img_out.shape[1], axis=1)

            for sensor_idx in range(img_out.shape[0]):
                for band_idx in range(img_out.shape[1]):
                    img_out[sensor_idx, band_idx, :, :] = convolve2d(img_out[sensor_idx, band_idx, :, :], PSF[sensor_idx, band_idx, :, :], mode='same', boundary='symm')

        # noise
        if config['NOISY'] and (SNR is not None):
            while SNR.ndim < img_out.ndim - 2:
                SNR = SNR[None, :]

            if SNR.shape[0] != img_out.shape[0]:
                SNR = np.repeat(SNR, img_out.shape[0], axis=0)

            if SNR.shape[1] != img_out.shape[1]:
                SNR = np.repeat(SNR, img_out.shape[1], axis=1)

            noise = np.empty(img_out.shape, dtype=float)

            for sensor_idx in range(img_out.shape[0]):
                for band_idx in range(img_out.shape[1]):
                    sigma = np.mean(img_raw[sensor_idx, band_idx, :, :][np.where(img_raw[sensor_idx, band_idx, :, :])]) / SNR[sensor_idx, band_idx]
                    noise[sensor_idx, band_idx, :, :] = np.random.normal(0, sigma, img_raw[sensor_idx, band_idx, :, :].shape)   # independent noise for each band

            img_out = np.clip((img_out.astype(float) + noise), 0, 255).astype(np.uint8)

        # preserve nodata entries
        img_out = np.where(img_raw, img_out, 0)
        return img_out


class Sensor:
    """
    Sensor object with own geometry and recorded bands.

    Parameters
    ----------
    name_str : string
        used as sensor name
    bands : tuple of int
        indices of recorded bands from input image file
    interior_orientation : Interior_orientation
        interior parameters of the camera
    pixels : int
        number of pixels in the line sensor
    px_size : float
        size of one pixel on the sensor
    SNRs : np.ndarray
        array containing signal-to-noise ratio of each band. Must have same length as bands

    Returns
    -------
    sensor : Sensor
        a new Sensor instance

    """

    def __init__(self, name_str, bands, interior_orientation, pixels, px_size, SNRs):
        self.name = name_str
        self.bands = bands
        self.SNRs = np.array(SNRs)
        if self.SNRs.size != len(self.bands) and self.SNRs.size != 1:
            raise ValueError('Specify one SNR per band, or one SNR to be used for all bands')
        self.interior_orientation = interior_orientation
        self.pixels = pixels
        self.px_size = px_size
        self.pose = None
        self.prev_pose = None
        self.prev_corners_XY = None
        self.px_corners_xy = self.get_px_corners()
        return

    def get_px_corners(self):
        """
        Compute location of pixel corners in image coordinates.

        Returns
        -------
        corners_xy : np.ndarray of float
            coordinates of all pixel corners, shape (2, self.pixels + 1, 1).
            corners_xy[0, :, :] contains x values, corners_xy[1, :, :] y values.
            Corners are sorted according to ascending x.

        """
        # horizontal positions of pixel corners in pixel coordinates
        px_cs = np.linspace(-.5, self.pixels + .5, self.pixels + 1)

        # vertical position of pixel corners in pixel coordinates (zero by definition)
        px_rs = np.zeros(px_cs.shape)

        # merge r and c values
        corners_rc = np.stack([px_rs, px_cs], axis=0)

        corners_xy = self.pixels_to_imgcoord(corners_rc)
        return corners_xy

    def pixels_to_imgcoord(self, coords_rc):
        """
        Compute image coordinates of a location in output image pixel coordinate system.

        Parameters
        ----------
        coords_rc : np.ndarray of float
            array of shape (2, ...), where coords_rc[0] contains only r coordinates, coords_rc[1] contains corresponding c coordinates.

        Returns
        -------
        coords_xy : np.ndarray of float
            array of shape coords_rc.shape, where coords_xy[0] contains only x coordinates, coords_xy[1] contains corresponding y coordinates.

        """
        # r = coords_rc[0]
        # c = coords_rc[1]

        x = (coords_rc[1] + 0.5) * self.px_size - self.px_size * self.pixels / 2
        y = - coords_rc[0] * self.px_size
        return np.array([x, y])

    def get_object_coordinates(self, img_coord, Z_terrain=0, pose=None):
        """
        Compute projection of image coordinates onto terrain in current pose.

        Parameters
        ----------
        img_coord : np.ndarray of float
            array of shape (2, ...), where img_coord[0] contains only x coordinates, img_coord[1] contains corresponding y coordinates.
        Z_terrain : float, optional
            vertical height of terrain at position (X,Y). The default is 0.
        pose : Cam_pose, optional
            pose representing exterior orientation to be used for calculation. If not given, current self.pose of the sensor is used.

        Returns
        -------
        coords_XY: np.ndarray of float
            array of shape img_coord.shape, where coords_XY[0] contains only X coordinates, coords_XY[1] contains corresponding Y coordinates.

        """
        if pose is None:
            pose = self.pose

        delta_x = self.interior_orientation.x0 - img_coord[0]
        delta_y = self.interior_orientation.y0 - img_coord[1]

        denominator = pose.R[2, 0] * delta_x + pose.R[2, 1] * delta_y + pose.R[2, 2] * self.interior_orientation.c
        const_factor = (Z_terrain - pose.XYZ_0[2]) / denominator

        X = pose.XYZ_0[0] + const_factor * \
            (pose.R[0, 0] * delta_x + pose.R[0, 1] * delta_y - pose.R[0, 2] * self.interior_orientation.c)
        Y = pose.XYZ_0[1] + const_factor * \
            (pose.R[1, 0] * delta_x + pose.R[1, 1] * delta_y - pose.R[1, 2] * self.interior_orientation.c)
        return np.stack([X, Y])

    def take_line_img(self, dop_processor, config=get_config(), feat_finder=None):
        """
        Acquire a single line image from current camera pose.

        Parameters
        ----------
        dop_processor : DOP_processor
            DOP_processor instance used to read data from DOP.
        config : Config, optional
            Dict of config parameters. If not given, default parameters are used.
        feat_finder : Feature_finder, optional
            Feature_finder instance used to detect GCPs.

        Returns
        -------
        img_out : np.array
            output images for all bands in self.bands, has shape (len(self.bands), 1, self.pixels).
        coverage : shapely.geoms.Polygon or MultiPolygon
            Polygon or collection of Polygons representing the ground track coverage.
         found_feats : list
             list of found GCPs.

        """
        img_out = np.zeros((len(self.bands), self.pixels), np.uint8)
        coverage_list = list()
        found_feats = list()

        new_corners_XY = self.get_object_coordinates(self.px_corners_xy, Z_terrain=config['MEAN_TERRAIN_HEIGHT'])
        if (not self.prev_pose == self.pose.previous) or (self.prev_corners_XY is None):
            if not self.pose.previous:
                return img_out, Polygon(), []
            self.prev_corners_XY = self.get_object_coordinates(self.px_corners_xy, Z_terrain=config['MEAN_TERRAIN_HEIGHT'], pose=self.pose.previous)

        for pixel in range(self.pixels):
            # pixel corners in object (XY) coordinates
            pixel_corners_XY = [tuple(new_corners_XY[:, pixel]),
                                tuple(self.prev_corners_XY[:, pixel]),
                                tuple(self.prev_corners_XY[:, pixel + 1]),
                                tuple(new_corners_XY[:, pixel + 1])]

            is_success, sample, new_coverage = dop_processor.sample_area(pixel_corners_XY, self.bands)

            if feat_finder:
                found_feats += feat_finder.check(new_coverage, pixel_corners_XY, self.pose.idx, pixel, self.name, self.pose)

            # update ground track coverage with new pixel area
            coverage_list.append(new_coverage)

            if not is_success:
                img_out[:, pixel] = 0   # write 0 (no data) to output image
                continue

            # take mean value of all contained DOP raster points (for each band) and write to output image
            img_out[:, pixel] = sample

        self.prev_pose = self.pose              # update previous pose
        self.prev_corners_XY = new_corners_XY   # update projected corner coordinates
        coverage = unary_union(coverage_list)
        return img_out, coverage, found_feats


class Interior_orientation:
    """
    Interior_orientation object to specify location of image principal point.

    Parameters
    ----------
    x0 : float
        principal point x offset in mm
    y0 : float
        principal point y offset in mm
    c : float
        camera focal length in mm

    Returns
    -------
    interior : Interior_orientation
        a new Interior_orientation instance.

    """

    def __init__(self, x0, y0, c):
        self.x0 = x0
        self.y0 = y0
        self.c = c
        return


class Cam_pose:
    """
    Cam_pose object to specify camera exterior orientation.

    Parameters
    ----------
    idx : int
        index of this pose in the orbit.
    coords_obj: tuple of float
        position of camera in object coordinates in (X0, Y0, Z0) format.
    attitude : tuple of float
        (alpha_y, alpha_x, alpha_z) angles from nominal orbit definition, in rad.
    deviate_angles: tuple of float, optional
        deviatory rotation angles from nominal orbit in (yaw, pitch, roll) format. Defaults to (0, 0, 0).

    Returns
    -------
    pose : Cam_pose
        a new Cam_pose instance.

    """
    # transformer from EPSG:25832 XYZ to ETRS89 lat/lon coordinates
    crs_obj = pyproj.CRS("epsg:25832")          # XYZ object coordinates
    lla = pyproj.CRS("epsg:4258")               # ETRS89 with GRS1980 ellipsoid
    obj_to_lla = pyproj.transformer.Transformer.from_crs(crs_obj, lla)

    # rotation matrix from xyz image to satellite vehicle coordinate axes
    R_img2sat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    def __init__(self, idx, coords_obj, attitude, deviate_angles=None):
        self.idx = idx
        self.XYZ_0 = np.array(coords_obj)
        self.previous = None
        alpha_y, alpha_x, alpha_z = attitude

        # rotation matrix from ECEF to EPSG:25832 object coordinates
        lat, lon, _ = Cam_pose.obj_to_lla.transform(self.XYZ_0[0], self.XYZ_0[1], self.XYZ_0[2], radians=True)
        R_ecef2obj = np.array([[-np.sin(lon), np.cos(lon), 0],
                               [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                               [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])

        # rotation matrix from satellite vehicle to ECEF coordinates
        R_sat2ecef = np.matmul(rotation_z(alpha_z), np.matmul(rotation_x(alpha_x), rotation_y(alpha_y)))

        if deviate_angles:
            # roll: "right wing down" positive
            # pitch: "nose up" positive
            # yaw: "turn right" positive
            # defined in satellite vehicle system
            yaw, pitch, roll = deviate_angles
            R_deviate = np.matmul(rotation_x(roll), np.matmul(rotation_y(pitch), rotation_z(yaw)))
        else:
            R_deviate = np.identity(3)

        # complete rotation matrix
        self.R = np.matmul(R_ecef2obj, np.matmul(R_sat2ecef, np.matmul(R_deviate, Cam_pose.R_img2sat)))
        return

    def __eq__(self, other):
        if isinstance(other, Cam_pose):
            return self.idx == other.idx and np.all(self.XYZ_0 == other.XYZ_0) and np.all(self.R == other.R)
        return False

    def copy(self):
        copy = object.__new__(Cam_pose)
        copy.idx = self.idx
        copy.previous = None
        copy.XYZ_0 = self.XYZ_0.copy()
        copy.R = self.R
        return copy

    def set_previous(self, prev_pose):
        self.previous = prev_pose.copy()
        return self
