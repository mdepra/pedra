import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import twirl

def solve_astrometry(img, pixel_scale=0.686 * u.arcsec,
                     ra_center_key='OBJCTRA', dec_center_key='OBJCTDEC'):
    """
    Perform astrometric solution on a FITS image and update the header with WCS information.

    Parameters
    ----------
    filepath (str)
        Path to the FITS file.
    
    pixel_scale (astropy.units.Quantity)
        Pixel scale of the detector in units of arcseconds per pixel.
    """

    # try:
    # Get the center of the image directly from the header
    ra = img.hdr[ra_center_key]
    dec = img.hdr[dec_center_key]
    center = SkyCoord(ra, dec, unit=(u.deg, u.deg))

    # Calculate the field of view size
    shape = img.shape
    fov = np.max(shape) * pixel_scale.to(u.deg)

    # Query GAIA stars in the field of view
    sky_coords = twirl.gaia_radecs(center, 1.2 * fov)[0:12]

    # Detect stars in the image
    pixel_coords = twirl.find_peaks(img.data)[0:12]

    # Compute the World Coordinate System (WCS)
    img.wcs = twirl.compute_wcs(pixel_coords, sky_coords)

    # Add WCS information to the header
    img.hdr.update(img.wcs.to_header())
    return img
    # except Exception as e:
    #     print(f"Error processing {img.label}: {e}")
    # finally:
    #     img.close() 