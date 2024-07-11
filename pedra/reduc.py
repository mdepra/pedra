r''' Routines to perform calibrations on fits images
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from .core import Image


def biascorrect(img, bias, prefix='b'):
    r''' Bias correction for an image.

    Parameters
    -----------
    img: pedra.Image or str
        Image to be bias-corrected.

    biasfile: pedra.Image or str
        Master bias image.

    prefix: str
        Prefix of the output image. Default is 'b'.

    Returns
    --------
    Bias-Corrected image.
    '''
    if isinstance(bias, str):
        # loading bias
        bias = Image(bias)
    if isinstance(img, str):
        # loading image
        img = Image(img)
    # removing bias
    bimg = img - bias
    bimg.label = prefix + bimg.label
    bimg.hdr['PEDRA_BIASCORR'] = True
    return bimg


def flatcorrect(img, flat, prefix='f'):
    r''' Flat field correction for an image.

    Parameters
    -----------
    img: pedra.Image or str
        Image to be bias-corrected.

    flatfile: str
        Master flat fits file


    prefix: str
        Prefix of the output image. Default is 'f'.

    Returns
    --------
    Flat-Divided image. 
    '''
    fimg = img / flat
    fimg.label = prefix + img.label
    fimg.hdr['PEDRA_FLATCORR'] = True
    return fimg


def combine(imglist, label='combine', metric='median', axis=0):
    r''' Combines a list of images using the mean or the median.

    Parameters
    -----------
    imglist: list
        List of fits file to be combined

    label: str
        Output file. If save=True

    metric: str
        Method of combination. Default is the mean.
        Options: Mean, Median

    Returns
    --------
    Combined Image object
    '''
    imgs_arr = [fits.data for fits in imglist]
    # applying combination method
    if metric == 'mean':
        img_combine = np.mean(imgs_arr, axis=axis)
    if metric == 'median':
        img_combine = np.median(imgs_arr, axis=axis)
    if metric == 'sum':
        img_combine = np.sum(imgs_arr, axis=axis)
    # producing new image
    img = Image(img_combine, imglist[0].hdr, wcs=imglist[0].wcs, label=label)
    img.hdr['PEDRA_COMBINE'] = 'Combined image'
    img.hdr['PEDRA_COMBINE_METRIC'] = metric
    return img
