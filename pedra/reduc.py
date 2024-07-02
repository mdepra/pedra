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
