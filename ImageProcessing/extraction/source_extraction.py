from astropy.io import fits
from astropy.wcs import WCS as AWCS
from astropy.table import Table, join, vstack
from photutils import DAOStarFinder, aperture_photometry, CircularAperture
from photutils.background import Background2D
from sklearn.cluster import DBSCAN
from mpdaf.obj import Image, WCS
from astropy.stats import sigma_clipped_stats
import numpy as np
import pylab as pl
import os


def plot_flux_aperture(tab1, tab2):
    pl.clf()
    sp = pl.subplot()
    p = np.where(tab2['aperture_sum_2'] > 45000)[0]
    sp.scatter(tab1['mag'][p], tab2['aperture_sum_1'][p])
    sp.scatter(tab1['mag'][p], tab2['aperture_sum_2'][p])
    sp.scatter(tab1['mag'][p], tab2['aperture_sum_3'][p])
    sp.set_yscale('log')
    pl.show()


def identify_sources(path):
    with fits.open(path) as fi:
        data = fi[0].data
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, iters=5)
        doafinder = DAOStarFinder(fwhm=4.2, threshold=5. * std)
        sources = doafinder(fi[0].data-median)
        wcs = AWCS(fi[0].header)
        ra, dec = wcs.all_pix2world(sources['xcentroid'],
                                    sources['ycentroid'],
                                    1)
        sources['ra'] = ra
        sources['dec'] = dec
        sources['jd'] = fi[0].header['JD-START']
        return sources


def aperature_photometry(sources, path):
    with fits.open(path) as fi:
        data = fi[0].data
        positions = []
        for s in sources:
            positions.append((s['xcentroid'], s['ycentroid']))
        radii = [3., 4., 7., 10.]
        apertures = [CircularAperture(positions, r=r) for r in radii]
        phot_table = aperture_photometry(data, apertures, method='subpixel',
                                         subpixels=5)
        sky = phot_table['aperture_sum_3']/radii[-1]**2
        sky -= phot_table['aperture_sum_2']/radii[-2]**2
        sky /= np.pi
        mag = phot_table['aperture_sum_1']/(radii[-3]**2*np.pi)-sky
        mag = -2.5*np.log10(mag)
        phot_table['mag'] = mag
        phot_table['id'] = sources['id']
        return phot_table


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


def companions(s1, s2):
    x = np.zeros((len(s1)+len(s2), 2))
    x[:len(s1), 0] = s1['ra']
    x[:len(s1), 1] = s1['dec']
    x[len(s1):, 0] = s2['ra']
    x[len(s1):, 1] = s2['dec']
    
    db = DBSCAN(eps=3./3600, min_samples=2)
    db.fit(x)
    
    s1['label'] = db.labels_[:len(s1)]
    s2['label'] = db.labels_[len(s1):]
    
    p = np.where(s1['label'] >= 0)[0]
    s1 = s1[p]
    
    p = np.where(s2['label'] >= 0)[0]
    s2 = s2[p]
    
    return join(s1, s2, keys='label')


if __name__ == '__main__':
    path = '/Users/patrickr/Documents/TEST/2017_10_13/'
    tot = []
    for f in os.listdir(path):
        if '_red' in f:
            print(f)
            sources = identify_sources(path+f)
            
            pm= np.where((sources['xcentroid'] > 10) &
                         (sources['ycentroid'] > 10) &
                         (sources['xcentroid'] < 1014) &
                         (sources['ycentroid'] < 1014))[0]
            sources = sources[p]
            comb = companions(sources, calib)
            fit, cov = np.polyfit(comb['mag'], comb['mag_med'], 2,
                             cov=True, w=1./comb['mag_std'])
            err = np.sqrt(np.diag(cov))
            err = np.poly1d(err)
            poly = np.poly1d(fit)
            
            err2 = poly.deriv(1)
            
            mag = sources['mag'].copy()
            sources['mag'] = poly(sources['mag'])
            sources['e_mag'] = err(np.abs(mag))
            sources['e_mag'] += abs(np.median(poly(comb['mag'])-comb['mag_med']))
            tot.append(sources)
    tot = vstack(tot)
    to = tot.copy()
    db = DBSCAN(eps=2./3600, min_samples=30)
    x = np.zeros((len(tot), 2))
    x[:, 0] = tot['ra']
    x[:, 1] = tot['dec']
    db.fit(x)
    tot['label'] = db.labels_
    p = np.where(tot['label'] >= 0)[0]
    tot = tot[p]
    
    tot = tot.group_by('label')
    med = tot.groups.aggregate(np.median)
    std = tot.groups.aggregate(np.std)
    cols = ['ra', 'dec', 'mag',# 'e_mag',
            'sharpness', 'roundness1', 'roundness2']
    for c in cols:
        med.rename_column(c, c+'_med')
        std.rename_column(c, c+'_std')
    
    stats = join(med[['ra_med', 'dec_med', 'mag_med', #'e_mag_med',
                      'label',
                      'sharpness_med', 'roundness1_med', 'roundness2_med']],
            std[['ra_std', 'dec_std', 'mag_std', #'e_mag_std',
                 'label',
                      'sharpness_std', 'roundness1_std', 'roundness2_std']],
                 keys='label')
    
    p = np.where((stats['mag_med'] < 0) &
                 (stats['mag_std'] > 0.15))[0]
    st = stats[p]
    pl.clf()
    sp = pl.subplot(211)
    sp2 = pl.subplot(212)
    jd0 = tot['jd'].min()
    for s in st:
        p = np.where(tot['label'] == s['label'])[0]
        lk = tot[p]
        lk['jd'] -= jd0
        lk.sort('jd')
        sp.errorbar(lk['jd'], lk['mag'], yerr=lk['e_mag'], fmt='.', 
                    label=str(s['label']))
        djd = []
        dmag = []
        for i, l in enumerate(lk):
            try:
                djd.extend(lk['jd'][i+1:]-l['jd'])
                dmag.extend(lk['mag'][i+1:]-l['mag'])
            except:
                pass
        sp2.hist(dmag, bins=30, histtype='step')
    sp.invert_yaxis()
    pl.legend(loc='best')
    
    fig = pl.figure(num=2)
    fig.clf()
    sp = fig.add_subplot(111)
    sp.scatter(stats['mag_med'], stats['mag_std'])
#    p = np.where((sources['xcentroid'] > 200) &
#                 (sources['ycentroid'] > 200))[0]
#    sources = sources[p]
#    phot_table = aperature_photometry(sources, path)
    # plot_flux_aperture(sources, phot_table)
#    print('psf photometry')
#    psf = psf_photometry(sources, path)
#    path = '/Users/patrickr/Documents/TEST/2017_10_13/ns142-V-0051_red.fits'
#    sources2 = identify_sources(path)
#    psf2 = psf_photometry(sources2, path)
#    p = np.where((psf['beta'] < 10) &
#                 (psf['fwhm_major'] > 2.5) &
#                 (psf['mag_err'] < 0.1) &
#                 (psf['beta'] < 5))[0]
#    psf = psf[p]

#    psf = join(sources, psf, keys='id')
#    ap = join(sources, phot_table, keys='id')
#    
#    pl.clf()
#    sp = pl.subplot(221)
#    sp.scatter(psf['mag_2'], psf['mag_1'])
#    sp = pl.subplot(222)
#    sp.scatter(psf['mag_2'], psf['beta'])
#    sp = pl.subplot(223)
#    sp.scatter(psf['mag_2'], psf['fwhm_major'])
#    sp = pl.subplot(224)
#    sp.scatter(psf['mag_2'], psf['mag_err'])
#    pl.show()

    # print(psf.show_in_browser(jsviewer=True,firefox