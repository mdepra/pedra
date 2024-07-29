r'''
'''
from datetime import datetime
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time


def moffat2D_integrate(amp, alpha, beta):
    """
    Computes the integral of multiple 2D Moffat functions.

    Parameters
    ----------
    params : list of float
        List of parameters for the Moffat functions.
        Each Moffat function is defined by 6 parameters:
        [amp, x0, y0, alpha, beta, offset].
    num_moffats : int
        Number of Moffat functions.

    Returns
    -------
    float
        Sum of integrals of the Moffat functions.
    """

    # if beta <= 1:
    #     raise ValueError("Beta must be greater than 1 for the integral to converge.")
    integral = (np.pi * amp * alpha**2) / (beta - 1)
    return integral

def moffat2D(xy, amp, x0, y0, alpha, beta, theta, offset):
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_rot = -sin_theta * (x - x0) + cos_theta * (y - y0)
    r2 = x_rot**2 + y_rot**2
    return amp * (1 + r2 / alpha**2)**(-beta) + offset


def multiple_moffats2D(xy, *params, 
                       return_flat=True, return_integral=False):
    """
    Computes the sum of multiple 2D Moffat functions.

    Parameters
    ----------
    xy : tuple of ndarray
        Coordinates (x, y) as arrays.
    *params : list of float
        List of parameters for the Moffat functions.
        Each Moffat function is defined by 6 parameters:
        [amp, x0, y0, alpha, beta, offset].

    Returns
    -------
    ndarray
        Computed values of the summed Moffat functions at each (x, y).
    """
    x, y = xy
    num_moffats = len(params) // 7
    result = np.zeros_like(x, dtype=float)
    for i in range(num_moffats):
        amp = params[7 * i]
        x0 = params[7 * i + 1]
        y0 = params[7 * i + 2]
        alpha = params[7 * i + 3]
        beta = params[7 * i + 4]
        theta = params[7 * i + 5]
        offset = params[7 * i + 6]
        result += moffat2D((x, y), amp, x0, y0, alpha, beta, theta, offset)

    if return_flat:
        result = result.ravel()
    if return_integral:
        integral = []
        for i in range(num_moffats):
            amp = params[7 * i]
            alpha = params[7 * i + 3]
            beta = params[7 * i + 4]
            integral.append(moffat2D_integrate(amp, alpha, beta))
        return result, integral
    else: 
        return result

def gaussian2D_integrate(amp, sigma_x, sigma_y):
    """
    Computes the integral of multiple 2D Gaussian functions.

    Parameters
    ----------

    Returns
    -------
    float
        Sum of integrals of the Gaussian functions.
    """

    integral = amp * 2 * np.pi * sigma_x * sigma_y 
    return integral

def gaussian2D(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    return amp * np.exp(-(a * (x - x0)**2 + 2*b * (x - x0) * (y - y0) + c * (y - y0)**2)) + offset


def multiple_gaussians2D(xy, *params, return_flat=True, return_integral=False):
    """
    Computes the sum of multiple 2D Gaussian functions.

    Parameters
    ----------
    xy : tuple of ndarray
        Coordinates (x, y) as arrays.
    *params : list of float
        List of parameters for the Gaussian functions.
        Each Gaussian function is defined by 6 parameters:
        [amp, x0, y0, sigma_x, sigma_y, offset].

    Returns
    -------
    ndarray
        Computed values of the summed Gaussian functions at each (x, y).
    """
    x, y = xy
    num_gaussians = len(params) // 7
    result = np.zeros_like(x, dtype=float)
    for i in range(num_gaussians):
        amp = params[7 * i]
        x0 = params[7 * i + 1]
        y0 = params[7 * i + 2]
        sigma_x = params[7 * i + 3]
        sigma_y = params[7 * i + 4]
        theta = params[7 * i + 5]
        offset = params[7 * i + 6]
        result += gaussian2D((x, y), amp, x0, y0, sigma_x, sigma_y, theta, offset)

    if return_flat:
        result = result.ravel()
    if return_integral:
        integral = []
        for i in range(num_gaussians):
            amp = params[7 * i]
            sigma_x = params[7 * i + 3]
            sigma_y = params[7 * i + 4]
            integral.append(gaussian2D_integrate(amp, sigma_x, sigma_y))
        return result, integral
    else: 
        return result


def smallbody_ephem(location='@hst',
                    target='CERES', date='2024-01-01', 
                    time='00:00:00.0', 
                    dateformat='%Y-%m-%d %H:%M:%S',
                    ephem=['RA', 'DEC', 'V']):
    r"""
    Astroquery wrapper function for obtaining a solar system small body ephemerides.

    Parameters
    ----------
    location: str
        JPL or MPC Observatory code.
    
    target: str
        Small body number, name or designation.
    
    date: str
        Date of asteroid observation.
    
    time: str
        Time of observation.
    
    dateformat: str
        Date and time formats. Default is '%Y-%m-%d %H:%M:%S'
    
    ephem: list
        Ephemeride keys that will be returned. 
        Full reference of supported values in:
        https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides  
    
    Returns
    -------
    Table with ephemerides
    """
    # Asteroid name
    name = target.strip().upper().replace('-', ' ')
    # Date and time of observation
    date = f"{date} {time}"
    date = datetime.strptime(date, dateformat)
    jd = Time(date).jd

    hor = Horizons(id=name,  location=location,
                    epochs=[jd])
    ephem_ = hor.ephemerides()
    if ephem is not None:
        ephem_ = ephem_[ephem]
    return ephem_


def smallbody_ephem_header(img, location='@hst',
                           targetkey='TARGNAME', datekey='DATE-OBS', 
                           timekey='TIME-OBS', exptimekey='EXPTIME', 
                           dateformat='%Y-%m-%d %H:%M:%S',
                           ephem=['RA', 'DEC', 'V']):
    r"""
    Astroquery wrapper function for obtaining a solar system small body ephemerides.

    Parameters
    ----------
    location: str
        JPL or MPC Observatory code.
    
    targetkey: str
        Header key for small body number, name or designation.
    
    datekey: str
        Header key for date of asteroid observation.
    
    timekey: str
        Header key for time of observation.
    
    dateformat: str
        Date and time formats. Default is '%Y-%m-%d %H:%M:%S'
    
    ephem: list
        Ephemeride keys that will be returned. 
        Full reference of supported values in:
        https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides  
    
    Returns
    -------
    Table with ephemerides
    """
    # Asteroid name
    name = img.hdr[targetkey].strip().upper().replace('-', ' ')
    # Time
    if timekey is None:
        date = f"{img.hdr[datekey].strip()}"
    else:
        date = f"{img.hdr[datekey].strip()} {img.hdr[timekey].strip()}"
    date = datetime.strptime(date, dateformat)
    jd = Time(date).jd

    hor = Horizons(id=name,  location=location,
                    epochs=[jd])
    ephem_ = hor.ephemerides()
    if ephem is not None:
        ephem_ = ephem_[ephem]
    return ephem_


def group_by_header(imglist, by='FILTER2'):
    r"""
    Group fits images by header values.
    
    Parameters
    ----------
    imglist: list
        List of PEDRA images to be sorted.
    """
    imgs_group = {}
    for img in imglist:
        if img.hdr[by] in imgs_group.keys():
            imgs_group[img.hdr[by]].append(img)
        else:
            imgs_group[img.hdr[by]] = [img]         
    return imgs_group


def sort_by_date(imglist, datekey='DATE-OBS', timekey='TIME-OBS', 
                 dateformat='%Y-%m-%d %H:%M:%S'):
    r"""
    Sort an image list based on header date of observation.

    Parameters
    ----------
    imglist: list
        List of PEDRA images to be sorted.
    
    datekey: str
        Header key for date of asteroid observation.

    timekey: str
        Header key for time of observation.
    
    dateformat: str
        Date and time formats. Default is '%Y-%m-%d %H:%M:%S'
    
    Returns
    -------
    Sorted list
    """
    jds = []
    for img in imglist:
        if timekey is None:
            date = f"{img.hdr[datekey].strip()}"
        else:
            date = f"{img.hdr[datekey].strip()} {img.hdr[timekey].strip()}"
            date = datetime.strptime(date, dateformat)
            jds.append(Time(date).jd)
    jds = np.array(jds)
    args = np.argsort(jds)
    imglist = [imglist[int(i)] for i in args]
    return imglist