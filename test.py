from ImageProcessing.astrometry import coordinate_align as CA
from ImageProcessing.extraction import source_extraction as SE
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from scipy.optimize import curve_fit
from astropy.table import vstack, join
from sklearn.cluster import DBSCAN
import os
import numpy as np
import math
Vizier.ROW_LIMIT =-1


def transform_axis(x, xf, y, center):
    fit, cov = curve_fit(lin_trafo,
                         xf, y)
    n_coordinates = lin_trafo(x, *fit)
    err = np.sqrt(np.diag(cov))
    return n_coordinates, [fit, err]


def transform_direct(x, x0, y0, ra0, dec0, a, b, c, d):
    y = x[len(x)//2:]-y0
    x = x[:len(x)//2]-x0
    
    xn = a*x+b*y+ra0
    yn = c*x+d*y+dec0
    return np.append(xn, yn)


def transform_direct2(x, x0, y0, ra0, dec0, c, ang):
    y = x[len(x)//2:]-y0
    x = x[:len(x)//2]-x0
    
    sini = math.sin(ang)
    cosi = math.cos(ang)
    
    xn = cosi*x-sini*y
    yn = sini*x+cosi*y
    
    xn *= c
    yn *= c
    
    xn += x0
    yn += y0
    return np.append(xn, yn)
    
    
def transform(x1, x2, x1f, x2f, y1, y2, x0, y0, fit_parameters=False):
    
    xf = np.append(x1f, x2f)
    x = np.append(x1, x2)
    ra, params_ra = transform_axis(x,
                                   xf,
                                   y1,
                                   (x0, y0))
    dec, params_dec = transform_axis(x,
                                     xf,
                                     y2,
                                     (x0, y0))
    
    if fit_parameters:
        return ra, dec, params_ra, params_dec
    return ra, dec


def poly_correction(xf, yf, x):
    fit = np.polyfit(xf, yf, 1)
    poly = np.poly1d(fit)
            
    return poly(x)


def polynomial_correction(comb, v, sources):
    v['ra'] = poly_correction(comb['ra'],
                              comb['xcentroid'],
                              v['ra'])
    comb = combine(sources, v)
    v['dec'] = poly_correction(comb['dec'],
                               comb['ycentroid'],
                               v['dec'])
    comb = combine(sources, v)
    v['ra'] = v['ra']-poly_correction(comb['dec'],
                                      comb['ra']-comb['xcentroid'],
                                      v['dec'])
    comb = combine(sources, v)
    v['dec'] = v['dec']-poly_correction(comb['dec'],
                                        comb['dec']-comb['ycentroid'],
                                        v['dec'])
    return v


def combine(sources, v):
        db = DBSCAN(eps=10, min_samples=2)
        x = np.zeros((len(sources)+len(v), 2))
        x[:len(v), 0] = v['ra']
        x[:len(v), 1] = v['dec']
        x[len(v):, 0] = sources['xcentroid']
        x[len(v):, 1] = sources['ycentroid']
        
        db.fit(x)
        
        v['label'] = db.labels_[:len(v)]
        sources['label'] = db.labels_[len(v):]
        pv = np.where(v['label'] >= 0)[0]
        ps = np.where(sources['label'] >= 0)[0]
        comb = join(v[['ra', 'dec', 'RAJ2000', 'DEJ2000', 'label']][pv],
                    sources[['xcentroid','ycentroid', 'label']][ps],
                    keys='label')
        return comb


def relative_pixel_coordinates(x, x_scale):
    xn = (x-np.min(x))/(np.max(x)-np.min(x))*x_scale
    return xn


def all_distances(x1, y1, x2, y2):
    x2 = np.array(x2)
    y2 = np.array(y2)
    xn = []
    yn = []
    for k, j in zip(x1, y1):
        xn.extend(k-x2)
        yn.extend(j-y2)
    return np.array(xn), np.array(yn)


def lin_trafo(x, a, b, c):
    y = x[len(x)//2:]
    x = x[:len(x)//2]
    
    n = a*x+b*y+c
    return n


def axis_shift(r):
    yh, ys = np.histogram(r, bins=100)
    p = np.where(yh == np.max(yh))[0][0]
    shift = ys[p]
    
    p = np.where(np.abs(r-shift) < 20)[0]
    shift = np.median(r[p])
    return shift


def shifts(x1, y1, x2, y2):
    xsa, rs = all_distances(x1,
                            y1,
                            x2,
                            y2)
    
    x_shift = axis_shift(xsa)
    y_shift = axis_shift(rs)
    
    return x_shift, y_shift


def identify_equa_componiens(sources, compare_catalog, x_col, y_col):
    db = DBSCAN(eps=2./3600, min_samples=2)
    x = np.zeros((len(sources)+len(compare_catalog), 2))
    x[:len(sources), 0] = sources['ra']
    x[:len(sources), 1] = sources['dec']
    x[len(sources):, 0] = compare_catalog[x_col]
    x[len(sources):, 1] = compare_catalog[y_col]
    db.fit(x)
    
    sources['label'] = db.labels_[:len(sources)]
    compare_catalog['label'] = db.labels_[len(sources):]
    
    p = np.where(sources['label'] >= 0)[0]
    s = sources[p]
    p = np.where(compare_catalog['label'] >= 0)[0]
    compare_catalog = compare_catalog[p]
#    print(s)
    cc = compare_catalog[[x_col, y_col, 'label']]
    s = s[['ra', 'dec', 'xcentroid', 'ycentroid', 'label']]
    comb = join(s,
                cc,
                keys='label')
    return comb


def precise_astrometry(sources, gaia):
    
    x0 = np.median(sources['xcentroid'])
    y0 = np.median(sources['ycentroid'])
    center = SkyCoord(np.median(sources['ra'])*u.deg,
                      np.median(sources['dec'])*u.deg)
    radius = np.hypot(sources['ra']-center.ra.degree,
                      sources['dec']-center.dec.degree)
    radius = np.max(radius)
#    print(len(sources))
    comb = identify_equa_componiens(sources, gaia, 'ra', 'dec')
    
#    print('')
#    print(len(comb))
    dra = comb['ra_1']-comb['ra_2']
    ddec = comb['dec_1']-comb['dec_2']
    dra *= 3600
    ddec *= 3600
    
#    print(round(np.mean(dra), 2), round(np.median(dra), 2),
#          round(np.std(dra), 2))
#    print(round(np.mean(ddec), 2), round(np.median(ddec), 2),
#          round(np.std(ddec), 2))
    print(x0, y0)
    ra, dec, param_ra, param_dec = transform(sources['xcentroid']-x0,
                                             sources['ycentroid']-y0,
                                             comb['xcentroid']-x0,
                                             comb['ycentroid']-y0,
                                             comb['ra_2'],
                                             comb['dec_2'],
                                             x0, y0,
                                             fit_parameters=True)
    return ra, dec, param_ra, param_dec, x0, y0


def astrometric_calibration(path, center, v, gaia):
    sources = SE.identify_sources(path)
    sources.sort('mag')
    sources_tot = sources.copy()
    sources = sources[:30]
    
    r = np.hypot(np.array(v['RAJ2000'])-center.ra.degree, 
                 np.array(v['DEJ2000'])-center.dec.degree)
    #        print(np.min(r))
    p = np.where(r == np.min(r))[0]
    c = v[p]
    p = np.where(v['R1mag'] <= c['R1mag'])[0]
    v = v[p]
    
    v.sort('R1mag')
#    v0 = v.copy()
    v = v[:30]
    
    v['ra'] = relative_pixel_coordinates(v['RAJ2000'], 1024)
    v['dec'] = relative_pixel_coordinates(v['DEJ2000'], 1024)

    x_shift, y_shift = shifts(sources['xcentroid'],
                              sources['ycentroid'],
                              v['ra'],
                              v['dec'])
            
    v['dec'] = v['dec']+y_shift
    v['ra'] = v['ra']+x_shift
    
    comb = combine(sources, v)
    
    p = np.where((comb['ra'] > 100))[0]
    comb = comb[p]
    
    v = polynomial_correction(comb, v, sources)
    comb = combine(sources, v)
    print(len(comb))
    if len(comb) < 10:
        
        return
    
    x0 = 512
    y0 = 512
            
    ra, dec, param_ra, param_dec = transform(sources_tot['xcentroid'],
                                             sources_tot['ycentroid'],
                                             comb['xcentroid'],
                                             comb['ycentroid'],
                                             comb['RAJ2000'],
                                             comb['DEJ2000'],
                                             x0, y0,
                                             fit_parameters=True)
    sources_tot['ra'] = ra
    sources_tot['dec'] = dec
    try:
        ra, dec, param_ra_gaia, param_dec_gaia, x0, y0 = precise_astrometry(sources_tot,
                                                                            gaia)
        param_ra = param_ra_gaia
        param_dec = param_dec_gaia
    except TypeError:
        pass
        
    ra_fit = param_ra[0]
    print(ra_fit)
    dec_fit = param_dec[0]
    cd = [[ra_fit[0], ra_fit[1]],
          [dec_fit[0], dec_fit[1]]]
    wcs = WCS()
    wcs.wcs.cd = cd
    wcs.wcs.crpix = [x0, y0]
#    print([x0, y0])
#    print([ra_fit[-1], dec_fit[-1]])
    wcs.wcs.crval = [ra_fit[-1], dec_fit[-1]]
#    print([ra_fit[-1], dec_fit[-1]])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
#    print(['RA---TAN', 'DEC--TAN'])
    
    ra1, dec1 = wcs.all_pix2world(sources_tot['xcentroid'],
                                  sources_tot['ycentroid'], 0)
#    print(np.median(np.hypot(ra-ra1, dec-dec1)*3600))
    wcs = wcs.to_header()
    with fits.open(path, mode='update') as fi:
        for h in wcs:
            fi[0].header[h] = wcs[h]
            
#    print(param_ra[0])
#    print(param_dec[0])
#    ra_fits.append(param_ra[0])
#    dec_fits.append(param_dec[0])


if __name__ == '__main__':
    path = '/Users/patrickr/Documents/TEST/2017_10_13/'
    s = SkyCoord('20h20m0.26s 4d37m55.07s')
    
    vizier = Vizier()
    vizier.ROW_LIMIT = -1
    v = vizier.query_region(s, 
                            radius=20*u.arcmin, 
                            catalog='USNO')
    #       print(v)
            
    v = v[-1]
    
    gaia = CA.get_calibration_sources(s, 20*u.arcmin)
    
    fail = 0
    count = 0
    ra_fits = []
    dec_fits = []
    so = []
    for f in os.listdir(path):
        if 'master' in f:
            continue
#        if 'ns142-V-0001.fit' not in f:
#            continue
        try:
            count += 1
            print(f)
            astrometric_calibration(path+f, s, v.copy(), gaia.copy())
            sources = SE.identify_sources(path+f)
#            p = np.where(sources['mag'] <= np.median(sources['mag']))[0]
#            sources = sources[p]
            with fits.open(path+f) as fi:
                wcs = WCS(fi[0].header)
                ra, dec = wcs.all_pix2world(sources['xcentroid'],
                                           sources['ycentroid'], 0)
                sources['ra'] = ra
                sources['dec'] = dec
                sources['jd'] = fi[0].header['JD-START']
                so.append(sources[['ra', 'dec', 'mag', 'jd']])
#            break
        except ValueError as e:
            fail += 1
            print(e)
    print(count, fail)
    so = vstack(so)
    db = DBSCAN(eps=5./3600, min_samples=10)
    x = np.zeros((len(so), 2))
    x[:, 0] = so['ra']
    x[:, 1] = so['dec']
    db.fit(x)
    so['id'] = db.labels_
    
    p = np.where(so['id'] >= 0)[0]
    so = so[p]
    
    so = so.group_by('id')
    print(so.groups.aggregate(np.mean))
    print(so.groups.aggregate(np.median))
    print(so.groups.aggregate(np.std))
#    