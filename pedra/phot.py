from .core import Image
from photutils import CircularAperture, aperture_photometry
import numpy as np

def extract_sources(img, nsource=10):
    r"""
    """
    img.data


def circular_photometry(img, x_center, y_center, radii):
    r"""
    Perform aperture photometry on the data.

    Parameters
    ----------
    img: pedra.Image
        The image to extract the photometry.

    x_center: float
        The x-coordinate of the center of the aperture.
    
    y_center: float
        The y-coordinate of the center of the aperture.
    
    radii: list of float
        Radii for the apertures.

    Returns
    -------
    tuple
        A tuple containing the photometry table and the apertures used.
    """
    positions = [(x_center, y_center)]
    if isinstance(radii, float):
        radii = [radii]
    apertures = [CircularAperture(positions, r=r) for r in radii]
    phot_table = aperture_photometry(img.data, apertures)
    return phot_table, apertures

def calculate_sky_background(img, sky_region):
    r"""
    Calculate the sky background statistics from the specified region.

    Parameters
    ----------
    img: pedra.Image
        The image to extract the photometry.

    sky_region: tuple
        A tuple specifying the region (x_start, x_end, y_start, y_end).

    Returns:
    tuple: A tuple containing the median, standard deviation, and mean of the sky background.
    """
    x_start, x_end, y_start, y_end = sky_region
    sky_data = img.data[y_start:y_end, x_start:x_end]
    sky_median = np.median(sky_data)
    sky_stdev = np.std(sky_data)
    sky_mean = np.mean(sky_data)
    return sky_median, sky_stdev, sky_mean