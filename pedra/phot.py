import numpy as np
from photutils import CircularAperture, aperture_photometry
#from copy import copy
#from .core import Image

class DataReduction:
    def __init__(self, data):
        """
        Initialize the DataReduction class with the given data.

        Parameters:
        data (numpy.ndarray): The data to be processed.
        """
        self.data = data
    
    def perform_aperture_photometry(self, x_center, y_center, radii):
        """
        Perform aperture photometry on the data.

        Parameters:
        x_center (float): The x-coordinate of the center of the aperture.
        y_center (float): The y-coordinate of the center of the aperture.
        radii (list of float): The list of radii for the apertures.

        Returns:
        tuple: A tuple containing the photometry table and the apertures used.
        """
        positions = [(x_center, y_center)]
        apertures = [CircularAperture(positions, r=r) for r in radii]
        phot_table = aperture_photometry(self.data, apertures)
        return phot_table, apertures

    def calculate_sky_background(self, sky_region):
        """
        Calculate the sky background statistics from the specified region.

        Parameters:
        sky_region (tuple): A tuple specifying the region (x_start, x_end, y_start, y_end).

        Returns:
        tuple: A tuple containing the median, standard deviation, and mean of the sky background.
        """
        x_start, x_end, y_start, y_end = sky_region
        sky_data = self.data[y_start:y_end, x_start:x_end]
        sky_median = np.median(sky_data)
        sky_stdev = np.std(sky_data)
        sky_mean = np.mean(sky_data)
        return sky_median, sky_stdev, sky_mean

    @staticmethod
    def calculate_magnitude(flux, zero_point=25, exposure_time=1):
        """
        Calculate the magnitude from the flux.

        Parameters:
        flux (float): The flux value.
        zero_point (float, optional): The zero-point magnitude. Default is 25.
        exposure_time (float, optional): The exposure time. Default is 1.

        Returns:
        float: The calculated magnitude.
        """
        return zero_point - 2.5 * np.log10(flux / exposure_time) if flux > 0 else float('nan')

    @staticmethod
    def calculate_magnitude_error(flux, flux_error):
        """
        Calculate the magnitude error from the flux and its error.

        Parameters:
        flux (float): The flux value.
        flux_error (float): The error in the flux value.

        Returns:
        float: The calculated magnitude error.
        """
        if flux > 0:
            return 2.5 / np.log(10) * (flux_error / flux)
        else:
            return float('inf')  # Return infinite for non-positive flux magnitudes

    @staticmethod
    def calculate_flux_error(flux, background, area, read_noise=9.9, gain=6.3):
        """
        Calculate the flux error considering various sources of noise.

        Parameters:
        flux (float): The flux value.
        background (float): The background value.
        area (float): The area of the aperture.
        read_noise (float): The read noise value.
        gain (float, optional): The gain value. Default is 1.0.

        Returns:
        float: The total calculated flux error.
        """
        signal_error = np.sqrt(flux / gain)
        background_error = np.sqrt(area * background / gain)
        read_noise_error = np.sqrt(area) * read_noise / gain
        total_error = np.sqrt(signal_error**2 + background_error**2 + read_noise_error**2)
        return total_error
