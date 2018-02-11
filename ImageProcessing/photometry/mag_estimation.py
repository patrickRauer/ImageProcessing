#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:23:07 2018

@author: Patrick Rauer
"""

from astropy.table import Table
# from matplotlib.colors import LogNorm
from sklearn.cluster import DBSCAN
# from photutils.background import Background2D as B2D
from astropy.modeling.models import Moffat2D, Gaussian2D, Polynomial2D
from astropy.modeling.fitting import LevMarLSQFitter
import numpy as np
import math
from mpdaf.obj import Image
import warnings
warnings.simplefilter('ignore')


def psf_photometry(sources, path, size=15):
    if True:
        image = Image(path)
        print('start photometry')
        psf = []
        for s in sources:
            x, y = s['ra'], s['dec']
            img = image.truncate(y-size/3600, y+size/3600,
                                 x-size/3600, x+size/3600)
            seg = img.segment()[0]
            fit = seg.moffat_fit(plot=False)
            psf.append((s['id'],
                        fit.center[0],
                        fit.center[1],
                        fit.err_center[0],
                        fit.err_center[1],
                        fit.flux,
                        fit.err_flux,
                        fit.peak,
                        fit.err_peak,
                        fit.fwhm[0],
                        fit.fwhm[1],
                        fit.err_fwhm[0],
                        fit.err_fwhm[1],
                        fit.n,
                        fit.err_n,
                        fit.rot,
                        fit.err_rot,
                        fit.cont,
                        fit.err_cont))
            print(seg)
        psf = Table(rows=psf,
                    names=['id',
                           'center_x',
                           'center_y',
                           'center_x_err',
                           'center_y_err',
                           'flux', 'flux_err',
                           'peak', 'peak_err',
                           'fwhm_major',
                           'fwhm_minor',
                           'fwhm_major_err',
                           'fwhm_minor_err',
                           'beta', 'beta_err',
                           'rotation', 'rotation_err',
                           'continuum', 'continuum_err'])

        psf['mag'] = -2.5*np.log10(psf['flux'])
        psf['mag_err'] = np.abs(2.5/(psf['flux']*np.log(10)))*psf['flux_err']
        return psf


def neighbour(k):
    out = []
    xa = k[1]
    ya = k[0]
    for x, y in zip(xa, ya):
        r = np.abs(xa-x)+np.abs(ya-y)
        
        if 1 in r:
            out.append((y, x))
    out = np.array(out)
    return out


def _pos(pos_sq, dp_noise, norm2, sq=True):
    err = np.sum(dp_noise*pos_sq)
    err /= norm2
    if sq:
        return math.sqrt(err)
    else:
        return err


def _positions(dp, pos):
    """
    Estimates the position of the source by the center of light

    :param dp: The light distribution of the source
    :type dp: numpy.ndarray
    :param pos: The coordinates of dp
    :type pos: tuple
    :return: The x and y coordinates, the square values of x and y and xy of the source
    :rtype: float, float, float, float, float
    """
    norm = np.sum(dp)
    
    x = _pos(pos[1], dp, norm, sq=False)
    y = _pos(pos[0], dp, norm, sq=False)
    x_sq = _pos(pos[1]**2, dp, norm, sq=False)
    y_sq = _pos(pos[0]**2, dp, norm, sq=False)
    xy = _pos(pos[1]*pos[0], dp, norm, sq=False)
    
    x_sq -= x**2
    y_sq -= y**2
    xy -= x*y
    
    return x, y, x_sq, y_sq, xy


def _noise(d, pos):
    """
    Estimates the background noise at the source location

    :param d: The complete image
    :type d: numpy.ndarray
    :param pos: The position of the source
    :type pos: tuple
    :return: The rms, median background around the source
    :rtype: float, float
    """
    xs, xe = max(np.min(pos[1])-10, 0), min(np.max(pos[1])+10, d.shape[1])
    ys, ye = max(np.min(pos[0])-10, 0), min(np.max(pos[0])+10, d.shape[0])

    d_noise = d[ys:ye, xs:xe]
    # _psf_moffat(d_noise)

    p_noise = (pos[0]-ys, pos[1]-xs)
    d_noise[p_noise] -= d_noise[p_noise]
    p = np.where(d_noise > 0)
    dn = d_noise[p]
    noise = np.std(dn)
    noise_med = np.median(dn)
    # p = np.where(d_noise >= noise_med)

    # p = (p[0]+ys, p[1]+xs)

    return noise, noise_med


def _psf_moffat(d):
    d = d-np.nanmedian(d)
    moffat = Gaussian2D(np.max(d), d.shape[1]/2, d.shape[0]/2)#+Polynomial2D(degree=0)
    fitter = LevMarLSQFitter()
    # print(d.shape)
    y, x = np.mgrid[:d.shape[0], :d.shape[1]]
    rs = fitter(moffat, x, y, d)
    cov = fitter.fit_info['cov_x']
    cov = fitter.fit_info['param_cov']
    # print(fitter.fit_info)
    err = np.sqrt(np.diag(cov))
    
    print(rs)
    print(err)
    print(round(100*err[0]/rs.amplitude.value))
    # print(rs.x_stddev, rs.y_stddev)


def _position_err(dp, noise, pos, x, y):
    """
    Estimates the error of the different position indicators

    :param dp: The light distribution of the sources
    :type dp: numpy.ndarray
    :param noise: The background noise at the source
    :type noise: float
    :param pos: The positions correspond to light distribution
    :type pos: tuple
    :param x: The center position in x-direction
    :type x: float
    :param y: The center position in y-direction
    :type y: float
    :return: The errors in x, y and xy direction
    :rtype: float, float, float
    """
    dp_noise = noise**2+dp/20
    norm2 = np.sum(dp_noise)**2
    
    x_err = _pos((pos[1]-x)**2, dp_noise, norm2)
    y_err = _pos((pos[0]-y)**2, dp_noise, norm2)
    xy_err = _pos((pos[1]-x)*(pos[0]-y), dp_noise, norm2)
    
    return x_err, y_err, xy_err


def _orientation(x_sq, y_sq, xy):
    """
    Estimates the orientation and the semi major and minor axis of the source

    :param x_sq: The square x coordinates
    :type x_sq: float
    :param y_sq: The square y coordinates
    :type y_sq: float
    :param xy: The xy coordinate
    :type xy: float
    :return: The semi-major, semi-minor axis and the rotation angle
    :rtype: float, float, float
    """
    theta = 2*xy/(x_sq+y_sq)
    theta = math.atan(theta)/2
    sq = math.sqrt(((x_sq-y_sq)/2)**2+xy**2)
    a = (x_sq+y_sq)/2+sq
    b = (x_sq+y_sq)/2-sq
    b = math.sqrt(b)
    a = math.sqrt(a)
    
    return a, b, theta
    

def center_of_light(d, pos, back=None):
    """
    Estimates the properties of the selected source (source at the position pos).

    :param d: The image
    :type d: numpy.ndarray
    :param pos: The position of the source on the image in coordinates of the image ndarray
    :type pos: tuple
    :param back: The background of the image with the same size as the image itself (not implemented now)
    :type back: numpy.ndarray
    :return:
        Returns the position of the source, the errors and the counts of the source
    :rtype: tuple
    """
    dp = np.float32(d[pos])

    # estimate the background noise
    noise, noise_med = _noise(d, pos)
    
    # TODO: NOISE ESTIMATION
    dp -= noise_med
    norm = np.sum(dp)

    # estimate the position data for this source
    x, y, x_sq, y_sq, xy = _positions(dp, pos)

    norm /= len(pos[1])

    # estimate the semi-major/minor axis and the rotation angle
    a, b, theta = _orientation(x_sq, y_sq, xy)
    
    sq = math.sqrt(((x_sq-y_sq)/2)**2+xy**2)
    cxx = y_sq/sq
    cyy = x_sq/sq
    cxy = -2*xy/sq

    # estimate the position errors
    x_err, y_err, xy_err = _position_err(dp, noise, pos, x, y)
    
    counts = np.sum(dp)
    iso_phot_correction = 1.-0.1961*noise_med/counts-0.7512*(noise_med/counts)**2
    iso_phot_correction = 2.5*np.log10(iso_phot_correction)
    # print(iso_phot_correction, counts)
    out = (x, y, x_sq, y_sq, norm, len(pos[1]), counts,
           noise, x_err, y_err, xy_err, cxx, cyy, cxy,
           a, b, theta, iso_phot_correction)
    return out


def source_detection(data, std_factor=0.35):
    """
    Estimates the parameters of all detected sources on the image

    :param data: The image
    :type data: numpy.ndarray
    :param std_factor: A scaling factor of the std of the image, used for the threshold estimation
    :type std_factor: float
    :return: Table with all detected sources, their positions and the magnitudes
    :rtype: astropy.table.Table
    """
    med = np.nanmedian(data)
    std = np.nanstd(data)
    threshold = med+std_factor*std
    
    pos = np.where(data >= threshold)
    x = np.zeros((len(pos[0]), 2))
    x[:, 0] = pos[0]
    x[:, 1] = pos[1]
    db = DBSCAN(eps=1., min_samples=5)
    db.fit(x)
    pdb = np.where(db.labels_ >= 0)[0]
    pos_y, pos_x = pos[0][pdb], pos[1][pdb]
    labels = db.labels_[pdb]
    
    positions = []

    # extract the data for all detected sources
    for i, l in enumerate(np.unique(labels)):
        if True:
            p = np.where(labels == l)[0]
            p_y = pos_y[p]
            p_x = pos_x[p]
            try:
                source = center_of_light(data, (p_y, p_x))
                positions.append(source)
            except ValueError:
                pass

    p = np.where(data < threshold)
    std = np.nanstd(data[p])
    med = np.nanmedian(data[p])
    return Table(rows=positions,
                 names=['xcentroid', 'ycentroid', 'x2', 'y2',
                        'sum', 'std_factor', 'A', 'noise',
                        'x_err', 'y_err', 'xy_err', 'cxx', 'cyy', 'cxy',
                        'a', 'b', 'theta', 'mag_corr']), med, std

        
def source_detection_w_error(data, std_factor=0.3):
    # std_factor = [0.4, 0.35, 0.3, 0.25, 0.2]
    sources, tot_med_noise, tot_std_noise = source_detection(data,
                                                             std_factor=std_factor)
    count_sum = np.array(sources['sum'])
    sources['err'] = 1.0857*np.sqrt(sources['A']*(sources['noise'])**2+count_sum/20)/count_sum
    sources['mag'] = -2.5*np.log10(count_sum)
    sources['mag_err'] = 2.5/(count_sum*np.log(10))*sources['err']
    sources['mag_err'] += 2.5/(tot_med_noise*np.log(10))*tot_std_noise
#    sources['mag'] += sources['mag_corr']
    sources['mag'] -= np.nanmin(sources['mag'])

    return sources
