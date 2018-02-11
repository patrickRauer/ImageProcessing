from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import join
from sklearn.cluster import DBSCAN
from .stats import AstrometryStats
from .catalogs import ExternalCatalog
from ImageProcessing.photometry.mag_estimation import source_detection_w_error
from .transformation_function import transform_axis
import numpy as np
Vizier.ROW_LIMIT = -1

__author__ = 'Patrick Rauer'
__version__ = '0.1'


def transform(x1, x2, x1f, x2f, y1, y2, fit_parameters=False, second=False):
    xf = np.append(x1f, x2f)
    x = np.append(x1, x2)
    ra, params_ra = transform_axis(x,
                                   xf,
                                   y1, second=second)
    dec, params_dec = transform_axis(x,
                                     xf,
                                     y2, second=second)

    if fit_parameters:
        return ra, dec, params_ra, params_dec
    return ra, dec


def poly_correction(xf, yf, x):
    """
    Estimates a 1. order polynomial correction of the data, based on the first both inputs and applies it to the third
    input.

    :param xf: The x-coordinates which are used to estimate the correction
    :type xf: numpy.ndarray
    :param yf: The y-coordinates which are used to estimate the correction
    :type yf: numpy.ndarray
    :param x: All x coordinates which should be transformed with the correction
    :type x: numpy.ndarray
    :return: The corrected x values
    :rtype: numpy.ndarray
    """
    fit = np.polyfit(xf, yf, 1)
    poly = np.poly1d(fit)
    return poly(x)


def polynomial_correction(comb, v, sources,
                          x1_col='xcentroid',
                          y1_col='ycentroid',
                          x2_col='ra',
                          y2_col='dec',
                          cols2=None):
    v[x2_col] = poly_correction(comb[x2_col],
                                comb[x1_col],
                                v[x2_col])
    comb = combine(sources, v, 
                   x1_col=x1_col,
                   y1_col=y1_col,
                   x2_col=x2_col,
                   y2_col=y2_col,
                   cols2=cols2,
                   eps=20)
    v[y2_col] = poly_correction(comb[y2_col],
                                comb[y1_col],
                                v[y2_col])
    comb = combine(sources, v, 
                   x1_col=x1_col,
                   y1_col=y1_col,
                   x2_col=x2_col,
                   y2_col=y2_col,
                   cols2=cols2,
                   eps=20)
    v[x2_col] = v[x2_col] - poly_correction(comb[y2_col],
                                            comb[x2_col] - comb[x1_col],
                                            v[y2_col])
    comb = combine(sources, v, 
                   x1_col=x1_col,
                   y1_col=y1_col,
                   x2_col=x2_col,
                   y2_col=y2_col,
                   cols2=cols2,
                   eps=20)
    v[y2_col] = v[y2_col] - poly_correction(comb[y2_col],
                                            comb[y2_col] - comb[y1_col],
                                            v[y2_col])
    return v


def combine(sources, v,
            x1_col='xcentroid',
            y1_col='ycentroid',
            x2_col='ra',
            y2_col='dec',
            cols1=None,
            cols2=None,
            eps=10.):
    """
    Combines two catalog into one with a DBSCAN algorithm

    :param sources: The first catalog
    :type sources: astropy.table.Table
    :param v: The second catalog
    :type v: astropy.table.Table
    :param x1_col: The name of the first column of the first catalog
    :type x1_col: str
    :param y1_col: The name of the second column of the first catalog
    :type y1_col: str
    :param x2_col: The name of the first column of the second catalog
    :type x2_col: str
    :param y2_col: The name of the second column of the second catalog
    :type y2_col: str
    :param cols1: A list with column names of the first catalog which should be included in the combined catalog
    :type cols1: list
    :param cols2: A list with column names of the second catalog which should be included in the combined catalog
    :type cols2: list
    :param eps: The maximal distance where two sources are the same
    :type eps: float
    :return: A combined catalog which includes sources which are in both catalogs
    :rtype: astropy.table.Table
    """
    if cols1 is None:
        cols1 = ['xcentroid', 'ycentroid', 'label']
    if 'label' not in cols1:
        cols1.append('label')
    if cols2 is None:
        cols2 = ['ra', 'dec', 'RAJ2000', 'DEJ2000', 'label']
    if 'label' not in cols2:
        cols2.append('label')
    l_v = len(v)
    db = DBSCAN(eps=eps, min_samples=2)
    x = np.zeros((len(sources) + l_v, 2))
    x[:l_v, 0] = v[x2_col]
    x[:l_v, 1] = v[y2_col]
    x[l_v:, 0] = sources[x1_col]
    x[l_v:, 1] = sources[y1_col]

    db.fit(x)

    v = exclude_sources(v, db.labels_[:l_v])
    sources = exclude_sources(sources, db.labels_[l_v:])
    comb = join(v[cols2],
                sources[cols1],
                keys='label')

    comb['c'] = 1.
    summ = comb[['label', 'c']].group_by('label')
    summ = summ.groups.aggregate(np.sum)
    p = np.where(summ['c'] == 1)[0]
    summ = summ[p]
    del comb['c']
    comb = join(comb, summ[['label']], keys='label')
    return comb


def all_distances(x1, y1, x2, y2):
    """
    Estimates all distances between x1, y1 and x2, y2 along x and y

    :param x1: The x-coordinates of the first data set
    :type x1: numpy.array
    :param y1: The y-coordinates of the first data set
    :type y1: numpy.array
    :param x2: The x-coordinates of the second data set
    :type x2: numpy.array
    :param y2: The y-coordinates of the second data set
    :type y2: numpy.array
    :return: The distances of both axises
    :rtype: numpy.array, numpy.array
    """
    x2 = np.array(x2)
    y2 = np.array(y2)
    xn = []
    yn = []
    for k, j in zip(x1, y1):
        xn.extend(k - x2)
        yn.extend(j - y2)
    return np.array(xn), np.array(yn)


def axis_shift(r):
    """
    Estimates the shift along one axis

    :param r: All possible distances along the axis
    :type r: numpy.ndarray
    :return: The shift and the standard derivation
    """
    yh, ys = np.histogram(r, bins=100)
    p = np.where(yh == np.max(yh))[0][0]
    shift = ys[p]

    p = np.where(np.abs(r - shift) < 20)[0]
    r = r[p]
    shift = np.median(r)
    return shift, np.std(r)


def shifts(x1, y1, x2, y2, err=False):
    """
    Estimates the shift between the coordinates along x and y axis

    :param x1: The first x-coordinates
    :type x1: numpy.array
    :param y1: The first y-coordinates
    :type y1: numpy.array
    :param x2: The second x-coordinates
    :type x2: numpy.array
    :param y2: The second y-coordinates
    :type y2: numpy.array
    :param err: True if the standard derivation should be included in the return values, else False (default)
    :type err: bool
    :return: The estimated shift along the x-axis and along the y-axis
    :rtype: float, float
    """
    xsa, rs = all_distances(x1,
                            y1,
                            x2,
                            y2)

    x_shift, x_std = axis_shift(xsa)
    y_shift, y_std = axis_shift(rs)

    if err:
        return (x_shift, x_std), (y_shift, y_std)
    return x_shift, y_shift


def exclude_sources(cat, labels):
    cat['label'] = labels
    p = np.where(labels > -1)[0]
    return cat[p]


def identify_equa_componiens(sources, compare_catalog, x_col, y_col):
    db = DBSCAN(eps=2. / 3600, min_samples=2)
    l_sources = len(sources)
    x = np.zeros((l_sources + len(compare_catalog), 2))
    x[:l_sources, 0] = sources['ra']
    x[:l_sources, 1] = sources['dec']
    x[l_sources:, 0] = compare_catalog[x_col]
    x[l_sources:, 1] = compare_catalog[y_col]
    db.fit(x)

    sources = exclude_sources(sources, db.labels_[:l_sources])
    compare_catalog = exclude_sources(compare_catalog, db.labels_[l_sources:])

    cc = compare_catalog[[x_col, y_col, 'label']]
    sources = sources[['ra', 'dec', 'xcentroid', 'ycentroid', 'label']]
    comb = join(sources,
                cc,
                keys='label')
    return comb


def center_coordinates(comb, col1_1, col1_2, col2_1, col2_2, x0, y0):
    """
    Finds the closest element to the center coordinates

    :param comb: The combined table
    :param col1_1: The name of the first column of the first catalog
    :type col1_1: str
    :param col1_2: The name of the second column of the first catalog
    :type col1_2: str
    :param col2_1: The name of the first column of the second catalog
    :type col2_1: str
    :param col2_2: The name of the second column of the second catalog
    :type col2_2: str
    :param x0: The central coordinates of the first column
    :type x0: float
    :param y0: The central coordinates of the second column
    :type y0: float
    :return: The new central coordinates of all four column
    :rtype: float, float, float, float
    """
    r = np.hypot(np.array(comb[col2_1])-x0,
                 np.array(comb[col2_2])-y0)
    p = np.where(r == np.min(r))[0][0]
    center = comb[p]

    x0, y0 = center[col1_1], center[col1_2]

    ra0, dec0 = center[col2_1], center[col2_2]

    return x0, y0, ra0, dec0


def precise_astrometry(sources, gaia, pixel_scale=2./3600):
    """
    Estimates a precise transformation between pixel coordinates and equatorial coordinates

    :param sources: The pre-transformed positions of sources on the image
    :type sources: astropy.table.Table
    :param gaia: A accurate astrometry catalog (standard is GAIA)
    :type gaia: astropy.table.Table
    :param pixel_scale: The pixel scale of the CCD chip in degree per pixel, default = 2/3600
    :type pixel_scale: float
    :return:
        The transformed coordinates (ra and dec), the fit parameters for both axis (fit values and errors)
        and the zero points (x and y)
    :type: numpy.ndarray, numpy.ndarray, list, list, float, float
    """
    x0 = np.median(sources['xcentroid'])
    y0 = np.median(sources['ycentroid'])

    comb = identify_equa_componiens(sources, gaia, 'ra', 'dec')

    x0, y0, ra0, dec0 = center_coordinates(comb,
                                           'xcentroid', 'ycentroid',
                                           'ra_2', 'dec_2', x0, y0)

    dra = comb['ra_1'] - comb['ra_2']
    ddec = comb['dec_1'] - comb['dec_2']
    dra *= 3600
    ddec *= 3600
    ra, dec, param_ra, param_dec = transform(np.array(sources['xcentroid']) - x0,
                                             np.array(sources['ycentroid']) - y0,
                                             np.array(comb['xcentroid']) - x0,
                                             np.array(comb['ycentroid']) - y0,
                                             np.array(comb['ra_2']-ra0)/pixel_scale,
                                             np.array(comb['dec_2']-dec0)/pixel_scale,
                                             fit_parameters=True,
                                             second=True)
    ra *= pixel_scale
    dec *= pixel_scale
    ra += ra0
    dec += dec0
    ra_fit = list(param_ra[0]*pixel_scale)
    dec_fit = list(param_dec[0]*pixel_scale)
    ra_fit.append(ra0)
    dec_fit.append(dec0)

    param_ra = [ra_fit, param_ra[1]]
    param_dec = [dec_fit, param_dec[1]]

    return ra, dec, param_ra, param_dec, x0, y0


def create_wcs(ra_fit, dec_fit, x0, y0):
    """
    Creates the WCS for the header of the image

    :param ra_fit: The transformation fit in RA direction
    :type ra_fit: list
    :param dec_fit: The transformation fit in DEC direction
    :type dec_fit: list
    :param x0: The origin of the x-coordinates
    :type x0: float
    :param y0: The origin of the y-coordinates
    :type y0: float
    :return: A WCS-object with the estimated data
    :rtype: astropy.wcs.WCS
    """
    cd = [[ra_fit[0], -ra_fit[1]],
          [dec_fit[0], -dec_fit[1]]]
    wcs = WCS(naxis=2)
    wcs.wcs.cd = cd
    wcs.wcs.crpix = [x0, y0]
    wcs.wcs.crval = [ra_fit[-1], dec_fit[-1]]
    wcs.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    wcs = WCS(wcs.to_header())
    return wcs


def copy_wcs(wcs):
    """
    Copies all attributes of the WCS's object, except the PC matrix.

    :param wcs: The WCS to copy
    :type wcs: astropy.wcs.WCS
    :return:
    """
    wcsn = WCS(naxis=2)
    wcsn.wcs.ctype = wcs.wcs.ctype
    wcsn.wcs.crval = wcs.wcs.crval
    wcsn.wcs.crpix = wcs.wcs.crpix
    return wcsn


def delete_wcs_header_entries(header):
    """
    Deletes all wcs header entries if there are any of them in it. If not nothing happens
    :param header: The header with old wcs entries
    :type header: astropy.io.fits.Header
    :return: The header without any wcs entries
    """
    # remove old WCS entries first
    c = ['WCSAXES', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
         'CDELT1', 'CDELT2', 'CUNIT1', 'CUNIT2', 'CTYPE1', 'CTYPE2', 'CRVAL1',
         'CRVAL2', 'LONPOLE', 'LATPOLE',
         'RADESYS', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
    for k in c:
        try:
            del header[k]
        except KeyError:
            pass

    return header


def add_wcs_header_entries(path, wcs):
    """
    Sets the new wcs system to the header of the fits-file

    :param path: The path to the file
    :type path: str
    :param wcs: The new wcs in a dict format
    :type wcs: astropy.io.fits.Header
    :return:
    """
    with fits.open(path, mode='update') as fi:
        fi[0].header = delete_wcs_header_entries(fi[0].header)
        # set the new WCS entries
        for h in wcs:
            fi[0].header[h] = wcs[h]


class Astrometry:

    path = ''
    image = None
    header = None
    sources = None
    sources_bright = None
    center_pixel = None
    center_image = None
    center_world = None
    wcs = None

    shifts = None

    usno = None
    gaia = None

    stats = AstrometryStats()
    comb = None

    def __init__(self, path, usno=None, gaia=None):
        self.path = path
        self.__load_image__()
        if usno is None:
            usno = ExternalCatalog('USNO', self.center_world)
            self.stats.add('external catalog', 'USNO-B loaded')
        if gaia is None:
            gaia = ExternalCatalog('GAIA', self.center_world)
            self.stats.add('external catalog', 'GAIA loaded')
        self.gaia = gaia
        self.usno = usno
        self.__get_coordinates__()

    def calibrate(self):
        """
        Performs the coordinate calibration

        :return:
        """
        self.stats.add('calibration', 'start')

        comb = self.__pixel_alignment__()
        self.__pixel2world__(comb)

    def __pixel_alignment__(self):
        """
        Aligns the coordinates in pixel like coordinate system

        :return:
        """

        cols2 = ['x_relative', 'y_relative', 'ra', 'dec', 'label']
        comb = self.__pixel_shift_correction__(cols2)

        comb = self.__pixel_poly_correction(comb, cols2)
        return comb

    def __pixel_poly_correction(self, comb, cols2):
        """
        Estimates and applies a polynomial correction between the systems

        :param comb: The combined catalog
        :type comb: astropy.table.Table
        :param cols2: A list with column names which should be included in the combined catalog
        :type cols2: list
        :return: The combined catalog
        :rtype: astropy.table.Table
        """
        self.usno.bright = polynomial_correction(comb,
                                                 self.usno.bright,
                                                 self.sources_bright,
                                                 x2_col='x_relative2',
                                                 y2_col='y_relative2',
                                                 cols2=cols2)
        self.stats.add('polynomial correction', 'done')
        comb = combine(self.sources_bright, self.usno.bright,
                       'xcentroid', 'ycentroid',
                       'x_relative2', 'y_relative2',
                       cols2=cols2,
                       eps=5)
        self.stats.add('2. combination', len(comb))
        self.__shift_stats__(comb)
        return comb

    def __pixel_shift_correction__(self, cols2):
        """
        Estimates and applies a shift correction between both coordinate systems

        :param cols2: A list with column names which should be included in the combined catalog
        :type cols2: list
        :return: The combined catalog
        :rtype: astropy.table.Table
        """
        # TODO: check if the x-pixel aren't inverted
        cols2.extend(['x_relative2', 'y_relative2'])
        x_shift, y_shift = shifts(self.sources_bright['xcentroid'],
                                  self.sources_bright['ycentroid'],
                                  self.usno.bright['x_relative'],
                                  self.usno.bright['y_relative'],
                                  err=True)
        self.stats.add('x-shift', x_shift)
        self.stats.add('y-shift', y_shift)
        self.shifts = (x_shift, y_shift)

        self.usno.bright['x_relative2'] = self.usno.bright['x_relative'] + x_shift[0]
        self.usno.bright['y_relative2'] = self.usno.bright['y_relative'] + y_shift[0]
        self.stats.add('align pixel', 'done')
        comb = combine(self.sources_bright, self.usno.bright,
                       'xcentroid', 'ycentroid',
                       'x_relative2', 'y_relative2',
                       cols2=cols2,
                       eps=20)
        self.stats.add('1. combination', len(comb))

        self.__shift_stats__(comb)
        return comb

    def __pixel2world__(self, comb):
        """
        Calculates the transformation between pixel coordinates and equatorial coordinates

        :param comb: First aligned coordinates
        :type comb: astropy.table.Table
        :return:
        """
        ra, dec, param_ra, param_dec = self.__usno_astrometry__(comb)

        try:
            param_ra, param_dec = self.__gaia_astrometry__()
        except TypeError as e:
            print(e)

        self.__correct_wcs__(param_ra, param_dec)

        wcs = self.wcs.to_header()

        # set the new WCS to the file
        add_wcs_header_entries(self.path, wcs)

        self.stats.add('WCS added to fits', 'done')
        self.stats.add('astronomical calibration', 'done')
        self.sources['jd'] = self.jd

        self.estimate_wcs_accuracies()

        self.set_equatorial_coordinates()

    def __usno_astrometry__(self, comb):
        """
        Performs the first transformation from pixel to equatorial coordinates

        :param comb: The combined catalog with pixel and equatorial coordinates
        :type comb: astropy.table.Table
        :return: The transformed coordinates (ra, dec) and the fit parameters (fit values and errors)
        :rtype: numpy.ndarray, numpy.ndarray, list, list
        """
        ra, dec, param_ra, param_dec = transform(self.sources['xcentroid'],
                                                 self.sources['ycentroid'],
                                                 comb['xcentroid'],
                                                 comb['ycentroid'],
                                                 comb['ra'],
                                                 comb['dec'],
                                                 fit_parameters=True)

        self.sources['ra'] = ra
        self.sources['dec'] = dec
        self.stats.add('pixel2equa', 'done')

        comb = self.__combine_with_sources__(self.usno.total)
        self.stats.add('3. combination', len(comb))
        self.__shift_stats__(comb,
                             x1_col='ra_1',
                             y1_col='dec_1',
                             x2_col='ra_2',
                             y2_col='dec_2',
                             x_name='RA (arcsec)',
                             y_name='DEC (arcsec)',
                             factor=3600)
        return ra, dec, param_ra, param_dec

    def __combine_with_sources__(self, external):
        """
        Combines a the source catalog with an external catalog

        :param external: The external catalog
        :type external: astropy.table.Table
        :return: The combined catalog
        :rtype: astropy.table.Table
        """
        comb = combine(self.sources, external,
                       'ra', 'dec', 'ra', 'dec',
                       cols1=['label', 'ra', 'dec', 'xcentroid', 'ycentroid'],
                       cols2=['label', 'ra', 'dec'],
                       eps=3./3600)
        return comb

    def __gaia_astrometry__(self):
        """
        Performs a accurate astrometry with gaia as reference catalog

        :return:
        """
        ra, dec, param_ra_gaia, param_dec_gaia, x0, y0 = precise_astrometry(self.sources,
                                                                            self.gaia.total)
        self.center_pixel = (x0, y0)
        self.sources['ra'] = ra
        self.sources['dec'] = dec
        self.stats.add('pix2equa (GAIA)', 'done')
        comb = self.__combine_with_sources__(self.gaia.total)
        self.stats.add('4. combination', len(comb))
        self.__shift_stats__(comb,
                             x1_col='ra_1',
                             y1_col='dec_1',
                             x2_col='ra_2',
                             y2_col='dec_2',
                             x_name='RA (arcsec)',
                             y_name='DEC (arcsec)',
                             factor=3600)
        self.comb = comb

        return param_ra_gaia, param_dec_gaia

    def __correct_wcs__(self, param_ra, param_dec):
        """
        Calculate and apply corrections of the WCS system

        :param param_ra: The fitting parameters for the transformation to RA
        :type param_ra: list
        :param param_dec: The fitting parameters for the transformation to Dec
        :type param_dec: list
        :return:
        """
        comb = self.comb
        ra_fit = param_ra[0]
        dec_fit = param_dec[0]

        self.wcs = create_wcs(ra_fit, dec_fit, self.center_pixel[0], self.center_pixel[1])
        self.__wcs_poly_correction__(comb, 'xcentroid', 'ycentroid', 'ra_2', 0)

        self.__wcs_poly_correction__(comb, 'xcentroid', 'ycentroid', 'dec_2', 1)

        self.set_equatorial_coordinates()

        # TODO: shift correction
        self.__wcs_shift_correction__()

        self.stats.add('WCS created', 'done')

    def __wcs_poly_correction__(self, comb, col1_1, col1_2, col2, index):
        """
        Estimates and applies the poly correction to RA or Dec

        :param comb: The combined data set of image sources and reference catalog sources
        :type comb: astropy.table.Table
        :param col1_1: The name of the first column in the image catalog
        :type col1_1: str
        :param col1_2: The name of the second column in the image catalog
        :type col1_2: str
        :param col2: The name of to transform axis
        :type col2: str
        :param index: The index of the to transform axis (RA=0, Dec=1)
        :type index: int
        :return:
        """
        fit = self.__wcs_poly_correction_fit__(comb, col1_1, col1_2, col2, index)

        wcs = copy_wcs(self.wcs)
        wcs.wcs.pc[index][index] = self.wcs.wcs.pc[index][index] * (1.-fit[0]/3600)

        cross = (index+1) % 2
        wcs.wcs.pc[index][cross] = self.wcs.wcs.pc[index][cross]
        wcs.wcs.pc[cross] = self.wcs.wcs.pc[cross]

        self.wcs = wcs

    def __wcs_shift_correction__(self):
        """
        Estimates and applies a shift to the WCS

        :return:
        """
        self.comb = self.__combine_with_sources__(self.gaia.total)

        dra_mean, dra_median, dra_std = print_coordinate_differences(self.comb, 'ra_1', 'ra_2', factor=3600.)
        ddec_mean, ddec_median, ddec_std = print_coordinate_differences(self.comb, 'dec_1', 'dec_2', factor=3600.)

        pix_scale = 2./3600
        self.wcs.wcs.cdelt = [pix_scale, pix_scale]
        self.wcs.wcs.pc /= pix_scale
        self.wcs.wcs.crval[0] += dra_median/3600
        self.wcs.wcs.crval[1] += ddec_median/3600

    def __wcs_poly_correction_fit__(self, comb, col1_1, col1_2, col2, index):
        """
        Estimates a polynomial correction of the WCS along one axis
        :param comb: The combined data set of sources on the image and in the reference catalog
        :type comb: astropy.table.Table
        :param col1_1: The name of the first column of the image sources
        :type col1_1: str
        :param col1_2: The name of the second column of the image sources
        :type col1_2: str
        :param col2: The name of the to transform column in the reference catalog
        :type col2: str
        :param index: The index of the reference column (0=RA and 1=Dec)
        :type index: int
        :return: The fitting values of the correction fit
        :rtype: list
        """
        ra, dec = self.wcs.all_pix2world(comb[col1_1], comb[col1_2], 1)
        column = ra
        if index == 1:
            column = dec
        fit = np.polyfit(comb[col2]-self.wcs.wcs.crval[index], (column-comb[col2])*3600, 1)
        return fit

    def set_equatorial_coordinates(self):
        """
        Transforms all pixel coordinates into equatorial coordinates
        :return:
        """
        ra, dec = self.wcs.all_pix2world(self.sources['xcentroid'], self.sources['ycentroid'], 1)
        self.sources['ra'] = ra
        self.sources['dec'] = dec

    def estimate_wcs_accuracies(self):
        """
        Estimates the accuracies of the WCS transformation
        :return:
        """
        wcs = WCS(self.path)
        ra, dec = wcs.all_pix2world(self.comb['xcentroid'], self.comb['ycentroid'], 1)
        self.comb['ra_2'] = ra
        self.comb['dec_2'] = dec
        dra = self.comb['ra_1']-self.comb['ra_2']
        dra *= 3600
        ddec = self.comb['dec_1']-self.comb['dec_2']
        ddec *= 3600

        print_coordinate_differences(self.comb, 'ra_1', 'ra_2', factor=3600.)
        print_coordinate_differences(self.comb, 'dec_1', 'dec_2', factor=3600.)

        self.comb['dra'] = dra
        self.comb['ddec'] = ddec
        fit, cov = np.polyfit(self.comb['dec_2'], self.comb['ddec'], 1, cov=True)
        print(fit, np.sqrt(np.diag(cov)))

    def __shift_stats__(self, comb,
                        x1_col='xcentroid',
                        y1_col='ycentroid',
                        x2_col='x_relative',
                        y2_col='y_relative',
                        x_name='x',
                        y_name='y',
                        factor=1.):
        """
        Estimates the shift stats of the combination

        :param comb: The combined catalog with coordinates from both catalogs
        :type comb: astropy.table.Table
        :param x1_col: The name of the first column of the first catalog
        :type x1_col: str
        :param y1_col: The name of the second column of the first catalog
        :type y1_col: str
        :param x2_col: The name of the first column of the second catalog
        :type x2_col: str
        :param y2_col: The name of the second column of the second catalog
        :type y2_col: str
        :param x_name: The x label for the stats
        :type x_name: str
        :param y_name: The y label fot the stats
        :type y_name: str
        :param factor: The multiplication factor for rescaling (like 3600 for degree to arcsec)
        :type factor: float
        :return:
        """
        dx_mean, dx_median, dx_std = coordinate_differences(comb, x1_col, x2_col, factor)
        dy_mean, dy_median, dy_std = coordinate_differences(comb, y1_col, y2_col, factor)

        self.stats.add(x_name+' offset', (dx_median, dx_std))
        self.stats.add(y_name+' offset', (dy_median, dy_std))

    def __get_coordinates__(self):
        """
        Load the sources from the image and split the 30 brightest ones into an additional variable

        :return:
        """
        self.stats.add('source extraction', 'start')
        self.sources = source_detection_w_error(self.image, 0.2)
        self.sources.sort('mag')
        self.sources_bright = self.sources[:30]
        self.stats.add('source extraction', 'done')
        self.stats.add('source No.', len(self.sources))

    def __load_image__(self):
        """
        Loads the image and initialize the corresponding variables

        :return:
        """
        with fits.open(self.path) as fi:
            self.image = fi[0].data
            self.header = fi[0].header
            self.center_pixel = (self.header['NAXIS1']/2,
                                 self.header['NAXIS2']/2)
            self.center_image = (self.header['NAXIS1']/2,
                                 self.header['NAXIS2']/2)
            ra = self.header['RA'].split(' ')
            dec = self.header['DEC'].split(' ')
            ra.extend(dec)
            self.center_world = SkyCoord('{}h{}m{}s {}d{}m{}s'.format(*ra))
            try:
                self.jd = self.header['JD-START']
            except KeyError:
                self.jd = 0
            self.stats.add('initialization', 'done')
            
    def evaluate_center(self):
        ra, dec = self.wcs.all_pix2world(self.center_image[0], 
                                         self.center_image[1],
                                         1)
        delta_ra = self.center_world.ra.degree-ra
        delta_dec = self.center_world.dec.degree-dec
        return delta_ra, delta_dec


def coordinate_differences(comb, col1, col2, factor=1.):
    """
    Calculates the differences between two columns

    :param comb: The combined catalog
    :type comb: astropy.table.Table
    :param col1: The name of the first column
    :type col1: str
    :param col2: The name of the second column
    :type col2: str
    :param factor: A scaling factor
    :type factor: float
    :return: Mean, median and standard derivation of the differences
    :rtype: float, float, float
    """
    delta_c1 = np.array(comb[col1]) - np.array(comb[col2])
    delta_c1 *= factor

    mean = np.mean(delta_c1)
    med = np.median(delta_c1)
    std = np.std(delta_c1)
    return mean, med, std


def print_coordinate_differences(comb, col1, col2, factor=1.):
    """
    Calculates the differences between two columns and print the results with two digits

    :param comb: The combined catalog
    :type comb: astropy.table.Table
    :param col1: The name of the first column
    :type col1: str
    :param col2: The name of the second column
    :type col2: str
    :param factor: A scaling factor
    :type factor: float
    :return: Mean, median and standard derivation of the differences
    :rtype: float, float, float
    """
    mean, med, std = coordinate_differences(comb, col1, col2, factor=factor)
    print(round(mean, 2),
          round(med, 2),
          round(std, 2))

    return mean, med, std
