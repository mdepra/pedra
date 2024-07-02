r'''
'''
from datetime import datetime
import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time


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
                           values=['RA', 'DEC', 'V']):
    r"""
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
    ephem = hor.ephemerides()
    if values is not None:
        ephem = ephem[values]
    return ephem