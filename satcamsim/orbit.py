"""Converts orbit data from .csv files to Cam_pose objects."""
import pyproj
import csv
import numpy as np
from os import listdir

import satcamsim.camera as camera
from .input_imgs import DOP_processor
from .support import get_config

# create coordinate reference systems and transformation between them
crs_obj = pyproj.CRS("epsg:25832")          # XYZ object coordinates
crs_ecef = pyproj.CRS("epsg:4936")          # ECEF
ecef_to_obj = pyproj.transformer.Transformer.from_crs(crs_ecef, crs_obj)


def get_pose_list(filename):
    """
    Compute all poses specified in a given orbit description.

    Parameters
    ----------
    filename : str
        path to .csv file containing the orbit information.

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

    # convert ECEF coordinates to object coordinates
    all_coords_obj = list(ecef_to_obj.itransform(all_coords_ECEF))
    all_coords_obj = [np.array(point) for point in all_coords_obj]

    print('Coordinate system conversion complete.')

    # convert to camera.Cam_pose objects
    pose_list = [camera.Cam_pose(idx, coords_obj, rotations) for idx, (coords_obj, rotations) in enumerate(zip(all_coords_obj, all_rotations))]
    for idx, pose in enumerate(pose_list):
        if idx == 0:
            continue
        pose.set_previous(pose_list[idx - 1])
    print('Cam_pose list complete.')
    return pose_list


def filter_poses(pose_list, camera, config=get_config()):
    """
    Filter poses such that only those showing parts of the DOP data are preserved.

    Parameters
    ----------
    pose_list : list[Cam_pose]
        list of poses to be filtered.
    camera : camera.Camera
        camera instance that will be used to simulate images. Must have sensors equipped.
    config : Config, optional
        Config parameters. Defaults are used, if not provided.

    Returns
    -------
    used_poses : list[Cam_pose]
        list containing poses relevant to the DOP data.

    """
    used_poses = []

    filelist = listdir(config['DOP_FOLDER'])

    with DOP_processor.from_config(config) as processor:
        for idx, pose in list(enumerate(pose_list))[::10]:
            camera.set_pose(pose)
            for sensor in camera.sensors:
                imgcoords = sensor.px_corners_xy[:, ::int(sensor.pixels / 10)]
                imgcoords = np.append(imgcoords, sensor.px_corners_xy[:, -1:], axis=1)
                objcoords = sensor.get_object_coordinates(imgcoords)
                num_points = objcoords.shape[1]
                for point in range(num_points):
                    file = processor.get_filenames(objcoords[:, point])
                    if file not in filelist:
                        continue
                    used_poses += pose_list[idx - 5:idx + 5]
                    break

    print('Filtering orbit complete.')
    return used_poses
