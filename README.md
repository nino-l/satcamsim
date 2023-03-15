# satcamsim: Satellite Camera Simulation

## Description

Image chain simulation is an essential tool for the design of Earth observation satellites. This Python package provides a simulator for the images produced by a low Earth orbit remote sensing nanosatellite, making use of projection and sampling models adjusted specifically to this application. The system's modulation transfer function is estimated according to analytical models. Images are simulated based on digital orthophotos and the satellite orbit.

## Installation

To install, clone the git repository to the desired location and set it as your working directory. To use the package from other locations, move the `satcamsim` directory to a location that will be found by your Python interpreter.

In addition to the standard library, `satcamsim` needs the following packages to be installed:

- [pyproj](https://github.com/pyproj4/pyproj) version 3.3.1
- [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) version 1.2.10
- [Shapely](https://shapely.readthedocs.io/en/1.8.2/manual.html) version 1.8.2

Other versions may work but have not been tested. Standard library dependencies are:
numpy,
itertools,
scipy,
multiprocessing,
os,
csv,
shutil,
uuid,
pickle,
PIL,
enum.

The package was developed on Python 3.9.7, but seems to run fine on Python 3.8.10 too.

## Usage

To get an idea of the usage of `satcamsim`, take a look at the `demo.py` script. An exemplary orbit section and a few ground control points are supplied in the corresponding subfolders. Note that neither orbit nor camera data resemble those of any existing or planned mission.

You will need to download the input DOP data (~5 GB) using the metalink file located at `demo/dop/dop40_GAP.meta4`, using for example the [DownThemAll browser extension](https://www.downthemall.net/). To prevent files from being downloaded twice, you may need to select a download server by filtering the URLs using `https://download1*` or `https://download2*` as wildcard expressions. The data is provided by Landesamt fuer Digitalisierung, Breitband und Vermessung Bayern under a [Creative Commons 4.0 license](https://creativecommons.org/licenses/by/4.0/deed.de).\
Save the files to `demo/dop/` or adjust `my_config['DOP_FOLDER']` to the correct path in `demo.py`.

It can be interesting to compare the simulated imagery with other sources. If you have one or multiple other files containing raster data, the script can cut the region corresponding to the satellite's ground coverage from them. To do so, you will have to adjust `my_config['COMPARE_FOLDER']` to these files' location, or move your files to the preset folder.

You should now be ready to run the script, which may take some time. After it has completed, the simulation results will be saved to `demo/output/`.

The `doc` directory contains more detailed documentation of submodules, classes, and functions.\
Start adjusting the default parameters in `configs.py` to best match you needs.

## Authors and acknowledgment

This project was developed by Nino Lenz as part of a bachelor's thesis, under supervision by Michael Greza, M.Sc., at the Chair of Photogrammetry and Remote Sensing, Technical University of Munich. A conference paper about the simulator is published as:\
Lenz, N., & Greza, M. (2023): Simulation of Earth-Observation Data Utilizing a Virtual Satellite Camera. Publikationen der Deutschen Gesellschaft fuer Photogrammetrie, Fernerkundung
und Geoinformation e.V., Wissenschaftlich-Technische Jahrestagung der DGPF, 22-23 March 2023, Munich.
