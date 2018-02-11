
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from astropy import units as u
import numpy as np
Vizier.ROW_LIMIT = -1


def relative_pixel_coordinates(x, x_0, y0):
    xn = (x - x_0) * 3600/2 + y0
    return xn


def get_usno_data(coordinates, radius=20*u.arcmin, mag_limit=16.):
    """
    Starts a query in the Vizier database to get the detection which are in the USNO-B catalog

    :param coordinates: The center coordinates
    :type coordinates: astropy.coordinates.SkyCoord
    :param radius: The search radius (default 20')
    :param mag_limit: The magnitude limit for the R1 magnitude (default 16)
    :type mag_limit: float
    :return: The reduced USNO-B catalog
    """
    v = Vizier.query_region(coordinates,
                            radius=radius,
                            catalog='USNO')
    v = v[-1]

    p = np.where(v['R1mag'] < mag_limit)[0]
    v = v[p]

    v.sort('R1mag')

    v.rename_column('RAJ2000', 'ra')
    v.rename_column('DEJ2000', 'dec')

    return v


def get_gaia_data(coordinates, radius=20*u.arcmin, mag_limit=16.):
    gaia_data = Gaia.cone_search(coordinates, radius)

    gaia_data = gaia_data.get_results()

    p = np.where(gaia_data['phot_g_mean_mag'] <= mag_limit)[0]
    gaia_data = gaia_data[p]

    gaia_data.sort('phot_g_mean_mag')

    return gaia_data


class ExternalCatalog:

    total = None
    bright = None
    radius = None
    image_size = None

    def __init__(self, catalog, center,
                 radius=20*u.arcmin,
                 brightness_limit=16,
                 no_bright_sources=30,
                 image_size=(1024, 1024)):
        self.radius = radius
        self.center = center
        self.image_size = image_size
        if catalog == 'USNO':
            self.total = get_usno_data(center, radius, brightness_limit)
        elif catalog == 'GAIA':
            self.total = get_gaia_data(center, radius, brightness_limit)
        self.relative_pixel_coordinates()
        self.bright = self.total[:no_bright_sources]

    def relative_pixel_coordinates(self):
        """
        Estimates of rough translation between RA and DEC coordinates and pixel coordinates

        :return:
        """
        self.total['x_relative'] = relative_pixel_coordinates(self.total['ra'],
                                                              self.center.ra.degree,
                                                              self.image_size[0]/2)
        self.total['y_relative'] = relative_pixel_coordinates(self.total['dec'],
                                                              self.center.dec.degree,
                                                              self.image_size[1]/2)
