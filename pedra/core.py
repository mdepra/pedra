
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from copy import copy
import rebin

from cana.util import kwargupdate

from astropy.io.fits.verify import VerifyWarning
import warnings
warnings.simplefilter('ignore', category=VerifyWarning)

import customtkinter as ctk

def loadimage_batch(imglist, data_ext=0, header_ext=0, wcs_ext=None,
                    err_ext=None, mask=None, labels=None, **kwargs):
    r"""
    Load a list images from file list. 

    Parameters
    ----------
    imglist: list
        List with Image files path.

    data_ext: int 
        Fits file extention for image data. 
        Default is 0. 

    header_ext: int 
        Fits file extention for header info. 
        Default is 0. 
         
    wcs_ext: None or int (Optional)
        Fits file extention for WCS info. 
        Default is None, which is for when WCS is not specified in the header.
        
    err_ext: None or int (Optional)
        Fits file extention for WCS info. 
        Default is None, which will not get an error array.
        
    mask: None or np.ndarray (Optional)
        Boolean numpy array for masking pixels in the image.
        Shape must be equal to data. Default is None.

    labels: list (Optional)
        List of images labels. 
        If not specified will get the base filenames as labels. 

    **kwargs: Accepts kwargs for Astropy.io.fits.open
    
    Returns
    -------
    PEDRA Image Object.    
    """
    if labels is not None:
        imgs = [loadimage(img, labels[i], data_ext=data_ext, header_ext=header_ext,
                          wcs_ext=wcs_ext, err_ext=err_ext, mask=mask, **kwargs) for i, img in enumerate(imglist)]
    else:
        imgs = [loadimage(img, data_ext=data_ext, header_ext=header_ext,
                          wcs_ext=wcs_ext, err_ext=err_ext, mask=mask, **kwargs) for img in imglist]
    return imgs


def loadimage(imgfile, data_ext=0, header_ext=0, wcs_ext=None,
              err_ext=None, mask=None, label=None, **kwargs): # -> implement method to merge images or headers
    r"""
    Load Image from file. 

    Parameters
    ----------
    imgfile: string
        Image file path.

    data_ext: int 
        Fits file extention for image data. 
        Default is 0. 

    header_ext: int 
        Fits file extention for header info. 
        Default is 0. 
         
    wcs_ext: None or int (Optional)
        Fits file extention for WCS info. 
        Default is None, which is for when WCS is not specified in the header.
        
    err_ext: None or int (Optional)
        Fits file extention for WCS info. 
        Default is None, which will not get an error array.
        
    mask: None or np.ndarray (Optional)
        Boolean numpy array for masking pixels in the image.
        Shape must be equal to data. Default is None.

    label: string (Optional)
        Image label. If not specified will get the base filename as label. 

    **kwargs: Accepts kwargs for Astropy.io.fits.open
    
    Returns
    -------
    PEDRA Image Object.    
    """
    # Label
    if label is None:
        label = os.path.basename(imgfile)
    # Header
    hdu = fits.open(imgfile, **kwargs)   
    hdr = hdu[header_ext].header 
    # Image data
    arr = np.array(hdu[data_ext].data, dtype=np.float64, order='F')
    # WCS
    if wcs_ext is not None:
        wcs_ext = fits.getheader(imgfile, ext=wcs_ext)
        wcs_ext = wcs.WCS(wcs_ext)
    # Error
    if err_ext is not None:
        err_ext = np.array(hdu[err_ext].data, dtype=np.float64, order='F')
    img = Image(arr, hdr, wcs_ext, err_ext, mask, label)
    return img


class Image(object):
    r"""
    Class to manipulate a fits Image file.
    """

    def __init__(self, data, hdr, wcs=None, err=None, mask=None, label='image'):
        r"""
        Initialize Image object.

        Parameters
        ----------
        data: np.ndarray
            The numpy array containing the data for the image.
        
        hdr: fits.header.Header
            Astropy FITS header object.
        
        wcs: wcs.WCS (Optional)
            Astropy WCS object with matrix for world coordinate system to/from pixel conversion.
        
        err: None or np.ndarray (Optional)
            The numpy array containing the error associated with the image data.
            Shape must be equal to data.

        mask: None or np.ndarray (Optional)
            Boolean numpy array for masking pixels in the image.
            Shape must be equal to data.

        label: string (Optional)
            Image label.
        """
        if err is not None:
            assert err.shape == data.shape
        if mask is not None:
            assert mask.shape == data.shape
        self.data = data
        self.hdr = hdr  
        self.label = label
        self.wcs = wcs
        self.err = err
        self.mask = mask

    @property
    def hdr_keys(self):
        r""" 
        List Header keys.
        """
        keys = list(set(self.hdr.keys()))
        return keys


    @property
    def centerpixel(self):
        r"""
        X,Y of pixel at center position. 
        """
        rownum = int(self.data.shape[0] / 2.0)
        colnum = int(self.data.shape[1] / 2.0)
        return (rownum, colnum)

    @property
    def centercoords(self):
        r"""
        X,Y of pixel at center position. 
        """
        assert self.wcs is not None, "WCS not available, try from_header method"
        ra, dec = list(np.array(self.wcs.pixel_to_world_values(*self.centerpixel)))   
        return (ra, dec)
    
    @property
    def shape(self):
        r"""
        Data array shape.
        """
        return self.data.shape

    def dist_from_center(self, row, col):
        r"""
        Distance from pixel to center in world coordinate angles. 
        If WCS=None, also returns None.

        Parameters
        ----------
        row: float
            X pixel value.
        
        col: float
            Y pixel value.

        Returns
        -------
        RA and DEC distances from the center.
        """
        if self.wcs is None:
            return None
        else:
            raise NotImplemented()  #### < need to implement this
        

    def pix_from_center(self, row, col):
        r"""
        Distance from pixel to center in pixels. 

        Parameters
        ----------
        row: float
            X pixel value.
        
        col: float
            Y pixel value.

        Returns
        -------
        X and Y distances from the center.
        """
        center_row, center_col = self.centerpixel
        delta_row = row - center_row
        delta_col = col - center_col
        return delta_row, delta_col



    def trim(self, tlims=[[50, 4110], [950, 1200]], prefix='t'):
        r"""
        Trim the image border.

        Parameters
        -----------
        tlims: list
            The limits of the trim region.. The format should be:
            [[<lower_x_lim>, <upper_x_lim>], [<lower_y_lim>, <upper_y_lim>]]

        prefix: str
            Prefix of the output image label. 

        Returns
        --------
        Trimmed Image object
        """
        # generate output
        img = copy(self)
        # cutting image array
        img.data = self.data[tlims[1][0]:tlims[1][1], tlims[0][0]:tlims[0][1]]
        # handlig label
        if prefix is not None:
            img.label = prefix + self.label
        img.hdr['PEDRA_TRIM'] = tlims.__repr__()
        # error array
        if self.err is not None:
            img.err = self.err[tlims[1][0]:tlims[1][1], tlims[0][0]:tlims[0][1]]
        # mask array
        if self.mask is not None:
            img.mask = self.mask[tlims[1][0]:tlims[1][1], tlims[0][0]:tlims[0][1]]
        return img

 
    def rebin(self, binsize=12,  prefix='r', func=np.sum): # -> implement copy img
        r"""
        Image rebinning.

        Parameters
        ----------
        binsize: integer
            Pixel size of the bin.
        
        prefix: string (Optional)
            Prefix of the output image label. 

        func: numpy function
            Function for the binning. E.g. np.sum, np.mean, np.median
            Default is np.sum
        
        Returns
        -------
        Binned Image
        """
        # generate output
        img = copy(self)
        # binning image array
        img.data = rebin.rebin(self.data, binsize, func=func)
        # handlig label
        img.hdr['PEDRA_REBIN'] = str(binsize)
        if prefix is not None:  
            img.label = prefix + img.label
        # binning error array
        if self.err is not None:
            img.err = rebin.rebin(self.err, binsize, func=func)
        # binning mask array
        ### ???
        return img

    def normalize(self, prefix='n', func=np.sum, axis=0): # -> implement other metrics, like mean and polynomial fitting
        r''' Normalizes flat field

        Parameters
        ----------
        outlabel: str
            Output file. If save=True

        save: boolean
            True if want to save image.

        func: int
            Order of the polynomial to fit and normalize flat field

        plotfit: boolean
            If want to display the fit on the flat

        Returns
        --------
        Normalized Image object
        '''
        # generate output
        img = copy(self)
        # normalizing image array
        aux = func(img.data, axis=axis)
        img.data = img.data / aux
        # handlig label
        img.hdr['PEDRA_Normalize'] = f'{func.__name__} on axis:{axis}'
        if prefix is not None:  
            img.label = prefix + img.label
        # binning error array
        if self.err is not None:
            img.err = img.err/ aux
        return img

    def save(self, outdir='./', outfile=None, data_ext=1, hdr_ext=0, wcs_ext=1):
        r''' Save fits image.

        Parameters
        ----------
        outfile: str
            Name of the new image file
        '''
        # Insert value in header to motify last change on file
        self.hdr['PEDRA_EDIT'] = str(datetime.datetime.now())
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU(self.data, header=self.hdr))
        if self.wcs is not None:
            hdu.append(fits.ImageHDU(self.data, header=self.wcs.to_header()))

        if outfile is None:
            outfile = self.label
        if outfile[-5:] != '.fits':
            outfile += '.fits'
        hdu.writeto(outdir+outfile, overwrite=True)

    def make_mask(self, val=0): # -> rever isso daqui
        r"""
        """
        # if (not mask_err) | (self.err is None):
        #     mask = np.mask_where(img.data == 0, img.data, copy=False)
        # else:
        #     mask = np.mask_where((img.data == 0) | (img.err == 0),
        #                          img.data, copy=False)
        # self.mask = mask

    def __sub__(self, value):
        img = copy(self)
        if isinstance(value, Image):
            value = value.data
        img.data = self.data - value
        return img

    def __add__(self, value):
        img = copy(self)
        if isinstance(value, Image):
            value = value.data
        img.data = self.data + value
        return img

    def __div__(self, value):
        img = copy(self)
        if isinstance(value, Image):
            assert len(self.data) == len(value.data)
            img.data = np.divide(self.data, value.data)
        else:
            img.data = self.data / value
        return img

    def __mul__(self, value):
        img = copy(self)
        if isinstance(value, Image):
            assert len(self.data) == len(value.data)
            img.data = np.multiply(self.data, value.data)
        else:
            img.data = self.data * value
        return img


    def view(self, ax=None, show=False,
                savefig=False, figdir='.', **kwargs):
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
            # setting default values for image plot with matplotlib
            kwargs_defaults = {'cmap': plt.cm.gray, 
                            'vmin': np.nanmean(self.data) - 5*np.nanstd(self.data),
                            'vmax': np.nanmean(self.data) + 5*np.nanstd(self.data),
                            'origin': 'lower'}
            kwargs = kwargupdate(kwargs_defaults, kwargs)
            print(kwargs['vmin'], kwargs['vmax'])
            # plotting image
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            im = ax.imshow(self.data, **kwargs)
            # outputing image
            if savefig:
                plt.savefig(figdir+self.name+'.png')
            if show:
                plt.show()
            return im

    def view_contour(self, ax=None, show=False,
                savefig=False, figdir='.', **kwargs):
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
            # setting default values for image plot with matplotlib
            kwargs_defaults = {'cmap': plt.cm.gray, 
                               'origin': 'lower'}
            kwargs = kwargupdate(kwargs_defaults, kwargs)
            # plotting image
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            ax.contourf(self.data, **kwargs)
            # outputing image
            if savefig:
                plt.savefig(figdir+self.name+'.png')
            if show:
                plt.show()

    def hdr_window(self):
        header_window = ctk.CTk()
        header_window.title("Header")
        header_window.geometry("600x720")
        # Create a Text widget to display text
        header_text = self.hdr.__repr__()
        header_widget = ctk.CTkTextbox(header_window, wrap="word")
        # Insert vext
        header_widget.insert("0.0", header_text)  
        header_widget.pack(padx=5, pady=5, fill='both') 
        header_window.mainloop()

    def close_window(self, root):
        root.destroy()

    def __repr__(self):
        # wcs
        if self.wcs is not None:
            wcs = True
        else: 
            wcs = False
        # error
        if self.err is not None:
            err = True
        else: 
            err = False      
        # mask
        if self.mask is not None:
            mask = True
        else: 
            mask = False
        return (f"Image label: {self.label} \n"
                f" Shape: {self.shape} \n"
                f" WCS: {wcs} \n"
                f" Error array: {err} \n"
                f" Mask: {mask} \n")