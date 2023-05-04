"""Converts orbit data from .csv files to Cam_pose objects."""
import pyproj
import csv
import numpy as np

import satcamsim.camera as camera
from .input_imgs import Input_processor
from .support import get_config


def get_pose_list(filename, config=get_config()):
    """
    Compute all poses specified in a given orbit description.

    Parameters
    ----------
    filename : str
        path to .csv file containing the orbit information.
    config : Config, optional
        Config parameters. Defaults are used, if not provided.

    Returns
    -------
    pose_list : list of camera.Cam_pose
        list of all poses within the area of interest in XYZ object coordinates.

    """
    # read data from orbit specification
    timestamps = []
    all_coords_ECEF = []
    all_rotations = []

    with open(filename) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            timestamps.append(float(row[0]))
            all_coords_ECEF.append(np.array(row[1:4]).astype(float))

            rotation_rad = np.array(row[4:7]).astype(float)
            all_rotations.append(rotation_rad)

    print('Reading orbit data from file complete.')
    
    crs_obj = config['CRS_IN']                  # XYZ object coordinates
    crs_ecef = config['CRS_ORBIT']              # orbit coordinates
    ecef_to_obj = pyproj.transformer.Transformer.from_crs(crs_ecef, crs_obj)

    # convert ECEF coordinates to object coordinates
    all_coords_obj = list(ecef_to_obj.itransform(all_coords_ECEF))
    all_coords_obj = [np.array(point) for point in all_coords_obj]

    print('Coordinate system conversion complete.')
    
    # transformer from object coordinates XYZ to lat/lon coordinates
    lla = pyproj.CRS("epsg:4258")               # geographic coordinates
    trans_obj_to_lla = pyproj.transformer.Transformer.from_crs(crs_obj, lla).transform

    # convert to camera.Cam_pose objects
    pose_list = [camera.Cam_pose(idx, coords_obj, rotations, trans_obj_to_lla) for idx, (coords_obj, rotations) in enumerate(zip(all_coords_obj, all_rotations))]
    for idx, pose in enumerate(pose_list):
        if idx == 0:
            continue
        pose.set_previous(pose_list[idx - 1])
    print('Cam_pose list complete.')
    return pose_list


def filter_poses(pose_list, camera, granularity=10, config=get_config()):
    """
    Filter poses such that only those showing parts of the input data are preserved.

    Parameters
    ----------
    pose_list : list[Cam_pose]
        list of poses to be filtered.
    camera : camera.Camera
        camera instance that will be used to simulate images. Must have sensors equipped.
    granularity : int, optional
        controls the precision of the filtering, must be even and >= 2.
        For `granularity=n`, only every (10 * n)-th pixel in every n-th pose is checked for intersection with the input images.
        The first and last pixels (sensor edges) of every n-th pose are always checked.
        Higher values speed up filtering, but may lead to up to `granularity//2` relevant poses being missed near the edges of the input images.
        The default is 10.
    config : Config, optional
        Config parameters. Defaults are used, if not provided.

    Returns
    -------
    used_poses : list[Cam_pose]
        list containing poses relevant to the input data.

    """
    if granularity % 2 or granularity < 2:
        raise ValueError(f'granularity must be even and >= 2. Got {granularity}')
    
    used_poses = []

    with Input_processor.from_config(config) as processor:
        
        for idx, pose in list(enumerate(pose_list))[::granularity]:      # check every n-th pose
            camera.set_pose(pose)
            
            for sensor in camera.sensors:
                imgcoords = sensor.px_corners_xy[:, ::(10 * granularity)]   # check every (10*n)-th pixel
                imgcoords = np.append(imgcoords, sensor.px_corners_xy[:, -1:], axis=1)
                objcoords = sensor.get_object_coordinates(imgcoords)
                num_points = objcoords.shape[1]
                for point in range(num_points):
                    if not processor.get_filenames(objcoords[:, point]):    # corresponding input image could not be found
                        continue
                    used_poses += pose_list[idx - granularity // 2:idx + granularity // 2]  # append previous (n//2) and next (n//2) poses
                    break

    print('Filtering orbit complete.')
    return used_poses
