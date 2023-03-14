"""Provides output and temp file handling utilities."""

import os
from shutil import rmtree
from uuid import uuid4
import pickle
import time
from PIL import Image
from csv import writer


class FolderExistsError(FileExistsError):
    """Raised if attempting to create a directory which already exists."""


class Temp_handler:
    """Handles saving, loading, and deletion of temp files. Ensures deletion of all files when exiting."""

    def __init__(self):
        self.files = []
        return

    def from_config(config):
        """
        Initialize Temp_handler with config parameters.

        Parameters
        ----------
        config : Config
            Config instance containing desired parameters.

        Returns
        -------
        handler : Temp_handler
            initialized according to config parameters.

        """
        handler = Temp_handler()
        handler.parent_folder = config['OUTPUT_FOLDER']
        return handler

    def __enter__(self):
        """
        Create temp directories.

        Raises
        ------
        FolderExistsError
            if a directory with the same name already exists.

        Returns
        -------
        self : Temp_handler

        """
        super_temp_folder = self.parent_folder + 'temp\\'
        if not os.path.isdir(super_temp_folder):
            os.mkdir(super_temp_folder)

        temp_ident = str(uuid4())
        self.temp_folder = self.parent_folder + 'temp\\' + temp_ident + '\\'

        try:
            os.mkdir(self.temp_folder)
        except FileExistsError:
            raise FolderExistsError('folder already exists: ' + self.temp_folder)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensure deletion of all temp files when exiting context manager.

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
        if exc_type == FolderExistsError:
            return

        if os.path.isdir(self.temp_folder):
            rmtree(self.temp_folder)
        return

    def save(self, obj):
        """
        Save an object to a temp file.

        Parameters
        ----------
        obj : object
            The object to be stored. Must be pickleable.

        Returns
        -------
        ident : str
            unique identifier to retrieve the object from memory.

        """
        ident = str(uuid4())
        if os.path.exists(self.temp_folder + ident):
            raise FileExistsError('file already exists: ' + self.temp_folder + ident)
        with open(self.temp_folder + ident, 'wb') as pfile:
            pickle.dump(obj, pfile)
        self.files.append(ident)
        return ident

    def load(self, ident):
        """
        Load an object stored in a temp file from memory.

        Parameters
        ----------
        ident : str
            the object's unique identifier returned when storing it.

        Returns
        -------
        obj : object
            the stored object.

        """
        with open(self.temp_folder + ident, 'rb') as pfile:
            return pickle.load(pfile)

    def load_all(self):
        """
        Load all objects stored in temp files by this Temp_handler.

        Returns
        -------
        objs: list
            contains all loaded objects in the order they were stored in.

        """
        return [self.load(ident) for ident in self.files]

    def remove(self, ident):
        """
        Delete a stored object from memory.

        Parameters
        ----------
        ident : str
            the object's unique identifier returned when storing it.

        Returns
        -------
        None.

        """
        os.remove(self.temp_folder + ident)
        self.files.remove(ident)
        return


class Output_saver:
    """Provides output handling utilities."""

    def from_config(config):
        """
        Initialize Output_saver with config parameters.

        Parameters
        ----------
        config : Config
            Config instance containing desired parameters.

        Returns
        -------
        saver : Output_saver
            initialized according to config parameters.

        """
        saver = Output_saver()
        saver.parent_folder = config['OUTPUT_FOLDER']
        saver.setup()
        return saver

    def setup(self):
        """
        Create required files and directories.

        Raises
        ------
        FolderExistsError
            Raised if directory already exists..

        Returns
        -------
        None.

        """
        ident = time.strftime('%Y-%m-%d_%H-%M-%S') + '_output\\'
        self.out_folder = self.parent_folder + ident
        try:
            os.mkdir(self.out_folder)
        except FileExistsError:
            raise FolderExistsError('folder already exists: ' + self.out_folder + str(self.file_counter) + '\\')

        self.logfile = self.out_folder + 'log.txt'
        with open(self.logfile, 'w') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': directory and log file created\n')

        self.file_counter = 1
        return

    def __enter__(self):
        """
        Open Output_saver context and write message to log file.

        Returns
        -------
        self : Output_saver

        """
        with open(self.logfile, 'a') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': entering output saver\n')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Write message to log file and leave Output_saver context.

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
        with open(self.logfile, 'a') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': exiting output saver\n')
        return

    def save_imgs(self, imgs):
        """
        Save an iterable containing images.

        Parameters
        ----------
        imgs : iterable
            Iterable containing the images to be saved.

        Returns
        -------
        None.

        """
        for img in imgs:
            self.save_img(img)
        return

    def save_img(self, img, subfolder='', filename=None):
        """
        Save one or multiple images, single or multi-channel, contained within a single array.
        Detailed behavior:
            - if img.ndim < 2: a warning is raised and img is pickled instead of saved as an image
            - if img.ndim == 2: img saved as single-channel image
            - if img.ndim == 3:
                - if img.shape[0] == 1: img saved as single-channel image
                - if img.shape[0] in (3, 4): img saved as RGB/RGBA image
                - else: individual channels are saved as single-channel images in a subdirectory
            - else: img[i] is passed into this function recursively, thus reducing img.ndim by one.
        Log message is recorded accordingly.

        Parameters
        ----------
        img : np.ndarray
            the array containing one or multiple images.
        subfolder : str, optional
            name of the subdirectory the image will bestored in, if . The default is ''.
        filename : str, optional
            file name of the stored image. Works only if img can be saved in a single file.
            If not provided, a number will be assigned as file name.

        Raises
        ------
        FolderExistsError
            if a subdirectory with the same name already exists.

        Returns
        -------
        None.

        """
        if img.ndim < 2:
            filename = 'failed_img_' + str(self.file_counter)
            self.save(img, filename=filename)
            raise Warning('img must have img.ndim>=2, was instead pickled as ' + filename)
            with open(self.logfile, 'a') as log:
                log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': array with ndim < 2  could not be stored as image, pickled instead: ' + filename + '\n')

        elif img.ndim == 2:
            self.save_img(img[None, :, :])

        elif img.ndim == 3:
            if img.shape[0] in (1, 3, 4):
                if not filename:
                    filename = str(self.file_counter) + ".tif"
                with open(self.logfile, 'a') as log:
                    log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': start saving ' + subfolder + filename + '\n')
                im = Image.fromarray(img.swapaxes(0, 2).swapaxes(0, 1))
                im.save(self.out_folder + subfolder + filename)
                self.file_counter += 1
                with open(self.logfile, 'a') as log:
                    log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': finished saving ' + subfolder + filename + '\n')
            else:
                with open(self.logfile, 'a') as log:
                    log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': array with no. of channels other than 1, 3, or 4 will be stored as individual images\n')
                try:
                    os.mkdir(self.out_folder + str(self.file_counter) + '\\')
                except FileExistsError:
                    raise FolderExistsError('folder already exists: ' + self.out_folder + str(self.file_counter) + '\\')
                [self.save_img(img[None, i, :, :], subfolder=str(self.file_counter) + '\\', filename=str(i)) for i in range(img.shape[0])]

        else:
            with open(self.logfile, 'a') as log:
                log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': array ndim > 3 will be stored as individual images\n')
            for i in range(img.shape[0]):
                self.save_img(img[i, :, :, :])
        return

    def save_config(self, config):
        """
        Save current config parameters in the log file. Parameters are read from the values in camera and input_imgs modules.

        Returns
        -------
        None.

        """
        with open(self.logfile, 'a') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': start saving config parameters\n')
            log.write(str(config) + '\n')
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': finished saving config parameters\n')
        return

    def save_csv(self, iterables, header=None, filename=None):
        """
        Save an iterable of iterables as a csv file.

        Parameters
        ----------
        iterables : Iterable[Iterable]
            an Iterable containing the rows to be written to file.
        header : Iterable, optional
            an Iterable optionally specifying column headers. Must have len(header) == len(iterables).
        filename : str, optional
            name of the produced file. If not provided, a number will be assigned as file name.

        Returns
        -------
        None.

        """
        if not filename:
            filename = str(self.file_counter)
        self.file_counter += 1

        with open(self.out_folder + filename, 'w', newline='') as file:
            csv_writer = writer(file)

            if header:
                csv_writer.writerow(header)

            csv_writer.writerows(iterables)
        return

    def save(self, obj, filename=None):
        """
        Save arbitrary objects.

        Parameters
        ----------
        obj : object
            The object to be stored. Must be pickleable.
        filename : str, optional
            file name of the stored object. If not provided, a number will be assigned as file name.

        Returns
        -------
        None.

        """
        if not filename:
            filename = str(self.file_counter)
        self.file_counter += 1

        with open(self.logfile, 'a') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': start saving ' + filename + '\n')
        with open(self.out_folder + filename, 'wb') as pfile:
            pickle.dump(obj, pfile)
        with open(self.logfile, 'a') as log:
            log.write(time.strftime('%Y-%m-%d_%H-%M-%S') + ': finished saving ' + filename + '\n')
