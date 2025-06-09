import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from cana.util import kwargupdate
from .core import Image
from .apps import ImageListViewer

def loadJWSTCube(cubefile, label=None):
    r"""
    """
    cube = fits.open(cubefile)
    if label is None:
        label = os.path.basename(cubefile).split('.')[0]
    return Cube(cube["SCI"].data, cube["SCI"].header, cube["ERR"].data, label)


# def makecube(imglist, header_index):
#     r"""
#     """
#     data = [img.data for img in imglist]

class Cube:

    def __init__(self, data, header, err=None, label=None):
        r"""
        """
        self.data = data
        self.hdr = header
        self.err = err
        self.label = label

        # self.start = self.hdr["CRVAL3"]
        # self.incr = self.hdr["CDELT3"]
        # x = np.linspace(0, data.shape[0], data.shape[0])
        # self.wav = self.start + self.incr * x
    
    @property
    def nslices(self):
        return self.data.shape[0]

    def select_slice(self, slice):
        r"""
        """
        err = self.err
        if err is not None:
            err = err[slice]
        return Image(self.data[slice], self.hdr, wcs=False, label=str(slice), err=err)

    def align_trim_slices(self, cube, centers, sampling_factor=4):
        r"""
        """
        new_cube = copy(self)
        new_cube_data = []
        for slc in range(self.nslices):
            oimg = cube.select_slice(slc)
            img = self.select_slice(slc)
            at_img = img.align_and_trim(oimg, centers[slc][0], centers[slc][1])
            new_cube_data.append(at_img.data)
        new_cube_data = np.stack(new_cube_data).T
        new_cube.data = new_cube_data
        return new_cube

    def rebin_slices(self, binsize):
        r"""
        """
        cube = copy(self)
        assert(isinstance(binsize, int) or isinstance(binsize, tuple))
        if isinstance(binsize, int):
            new_cube_data = np.empty(shape=(cube.data.shape[0], 
                                            int(cube.data.shape[1]/binsize), 
                                            int(cube.data.shape[2]/binsize)))
        elif isinstance(binsize, tuple):
            new_cube_data = np.empty(shape=(cube.data.shape[0], 
                                            int(cube.data.shape[1]/binsize[0]), 
                                            int(cube.data.shape[2]/binsize[1])))
        for i in range(cube.nslices):
                    slc = cube.select_slice(i)
                    slc = slc.rebin(binsize)
                    new_cube_data[i]=slc.data
                    # print(slc.data.shape)
        cube.data = new_cube_data
        return cube

    def normalize_slices(self,  metric='median', axis=0):
        r"""
        """
        cube = copy(self)
        for i in range(cube.nslices):
            slc = cube.select_slice(i)
            slc = slc.normalize(metric=metric, axis=axis)
            cube.data[i]=slc.data
        return cube

    def stack_slices(self, size=10, start=0, end=None, method='median', axis=0):
        r"""
        """
        cube_new = copy(self)
        if end is None:
            end = self.data.shape[0]
        if method == 'median':
            func = np.median
            std_func = np.std  # ->> change later to mad
        if method == 'mean':
            func = np.mean
            std_func = np.std
        if method == 'sum':
            func = np.sum
        bins = np.arange(start, end, size)
        cube_new.data = np.array([func(self.data[bins[i]:bins[i+1]], axis=axis) for i, ii in enumerate(bins[:-1])])
        if self.err is not None:
            cube_new.err =  np.array([func(self.err[i:i+1], axis=axis) for i in bins]) #->> change this to actual error propagation
        return cube_new

    def view(self, ax=None, fig=None,
             show=False, savefig=False, figdir='.',
             label_kwargs=None,
              **kwargs):
            r''' Display image

            Parameters
            ----------
            fax (Optional): None, matplotlib.axes
                If desired to subplot image in a figure. Default is 'None', which
                will open a new plt.figure()

            show (Optional): boolean
                True if want to plt.show(). Default is True.

            savefig (Optional): boolean
                True if want to save image.

            figdir (Optional): str
                Only needed if savefig=True. Directory for saving image.
                The file basename will be the same name of the image fitsfile.
                The image extention is .png.

            **kwargs: matplotlib kwargs

            ''' 
            viewer = ImageListViewer(wcs=False, 
                                    cardinal=None,
                                    sundirection=False)
            imgs = [self.select_slice(idx) for idx in range(self.nslices)]
            viewer(imgs, 
                   label_kwargs=label_kwargs,
                   **kwargs)

            if savefig:
                plt.savefig(figdir+self.name+'.png')
            if show:
                plt.show()
            # return fig, ax

    def fitcube(self, init_x, init_y, fit_type='moffat'):
        r"""
        """

    def __sub__(self, val):
        new_cube = copy(self)
        if isinstance(val, self.__class__):
            val = val.data
        elif isinstance(val, np.ndarray):
            if self.data.shape[0] == val.shape[0]:
                val = np.broadcast_to(val, self.data.T.shape).T
        new_cube.data = self.data - val
        return new_cube
    
    def __add__(self, val):
        new_cube = copy(self)
        if isinstance(val, self.__class__):
            val = val.data
        elif isinstance(val, np.ndarray):
            if self.data.shape[0] == val.shape[0]:
                val = np.broadcast_to(val, self.data.T.shape).T
        new_cube.data = self.data + val
        return new_cube

    def __truediv__(self, val):
        new_cube = copy(self)
        if isinstance(val, self.__class__):
            val = val.data
            new_cube.data
        elif isinstance(val, np.ndarray):
            if self.data.shape[0] == val.shape[0]:
                val = np.broadcast_to(val, self.data.T.shape).T
        new_cube.data = self.data / val
        return new_cube

    def __mul__(self, val):
        new_cube = copy(self)
        if isinstance(val, self.__class__):
            val = val.data
        elif isinstance(val, np.ndarray):
            if self.data.shape[0] == val.shape[0]:
                val = np.broadcast_to(val, self.data.T.shape).T
        new_cube.data = self.data * val
        return new_cube