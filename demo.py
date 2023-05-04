"""While exemplary orbit data and GCPs come with this demo, input data and - if desired - comparison raster data have to be supplied as described in the README."""
import satcamsim as sim


def main():
    # create interior orientation instance
    rgb_interior = sim.Interior_orientation(0, 0, .65)

    # create sensor instance
    my_sensor = sim.Sensor('RGB', (1, 2, 3), rgb_interior, 4096, 6e-6, (90, 100, 110))

    # get filter mask to apply PSF
    psf = sim.mtf.get_PSF()
    psf = sim.mtf.trim_PSF(psf, 50)     # trim mask to desired size to speed up convolution

    # create camera instance and add sensor to it
    my_camera = sim.Camera()
    my_camera.add_sensor(my_sensor)

    # create config from default parameters specified in configs.py
    my_config = sim.get_config()

    # manually adjust some config parameters
    my_config['FOLDER_IN'] = './demo/dop/'
    my_config['COMPARE_FOLDER'] = './demo/comp/'

    my_config['OUTPUT_FOLDER'] = './demo/output/'

    my_config['FEATURE_CSV_PATH_2D'] = './demo/gcps/2D_Zugspitze.csv'
    my_config['FEATURE_CSV_PATH_3D'] = './demo/gcps/3D_Zugspitze.csv'
    my_config['MEAN_TERRAIN_HEIGHT'] = 970

    # read poses from .csv orbit description
    all_poses = sim.get_pose_list('./demo/orbits/orbit.csv')
    # only keep poses for which input files exist
    my_poses = sim.filter_poses(all_poses, my_camera, config=my_config)
    
    # only keep a few hundred line images to speed up demo - delete the next line to simulate the full swath (4630 lines for the DOP supplied with the metalink file)
    my_poses = [pose for pose in my_poses if (pose.idx >= 13800 and pose.idx < 14200)]

    with sim.Output_saver.from_config(my_config) as out:
        out.save_config(my_config)                      # write config parameters to logfile
        out.save(my_poses, filename='orbit_filtered.pkl')   # pickle pose list for later reference

        # simulate the image swath
        img_out, coverage, found_gcps, img_raw = my_camera.simulate_swath(my_poses, PSF=psf, config=my_config)

        out.save_img(img_out)     # save degraded image
        out.save_img(img_raw)     # save non-degraded image

        # save results of GCP localization as .csv
        out.save_csv(found_gcps, header=('GCP_ID', 'row_out (line_index)', 'col_out (pixel_index)', 'sensor_name', 'point_X', 'point_Y', 'point_Z'), filename='found_GCPs.csv')

        with sim.Comp_processor.from_config(my_config, folder_out=out.out_folder) as raster:
            raster.cut_geom(coverage)
    return


if __name__ == '__main__':
    main()
