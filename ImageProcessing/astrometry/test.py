from coordinate_align import Astrometry
from astropy.coordinates import SkyCoord
from astropy.table import join
from astropy import units as u
from sklearn.cluster import DBSCAN
from astroquery.vizier import Vizier
#from ImageProcessing.extraction.source_extraction import identify_sources
import pylab as pl
import numpy as np
import warnings
import os
from astropy.io import fits
from astropy.wcs import WCS as AWCS
from astropy.table import Table, join, vstack
from photutils import DAOStarFinder, aperture_photometry, CircularAperture

from astropy.stats import sigma_clipped_stats
warnings.simplefilter('ignore')
Vizier.ROW_LIMIT = -1
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
#        sources['jd'] = fi[0].header['JD-START']
        return sources
    
def combine_w_apass(sources, apass):
    db = DBSCAN(eps=5./3600, min_samples=2)
    x = np.zeros((len(sources)+len(apass), 2))
    x[:len(sources), 0] = sources['ra']
    x[:len(sources), 1] = sources['dec']
    x[len(sources):, 0] = apass['RAJ2000']
    x[len(sources):, 1] = apass['DEJ2000']
    
    db.fit(x)
    sources['label'] = db.labels_[:len(sources)]
    apass['label'] = db.labels_[len(sources):]
    p = np.where(sources['label'] >= 0)[0]
    sources = sources[p]
    p = np.where(apass['label'] >= 0)[0]
    apass = apass[p]
    
    comb = join(sources, apass, keys='label')
    return comb

def test_wcs(wcs, data):
    print("WCS TEST")
    print(wcs)
    print(len(data))
    data = data.group_by('label')
    data['c'] = 1.
    summ = data[['label', 'c']].groups.aggregate(np.sum)
    equa0 = wcs.all_pix2world(wcs.wcs.crpix[0],
                              wcs.wcs.crpix[1],
                              1)
    delta = equa0-wcs.wcs.crval
    delta *= 3600
    
    dra = (data['ra_1']-data['ra_2'])*3600
    ddec = (data['dec_1']-data['dec_2'])*3600
    print(round(np.median(dra), 2), round(np.mean(dra), 2), round(np.std(dra), 2))
    print(round(np.median(ddec), 2), round(np.mean(ddec), 2), round(np.std(ddec), 2))
    return delta

def astrometric_checking():
    dra = (astrometry.comb['ra_1']-astrometry.comb['ra_2'])*3600
    ddec = (astrometry.comb['dec_1']-astrometry.comb['dec_2'])*3600
    pl.clf()
    sp = pl.subplot(221)
    sp.scatter(astrometry.comb['ra_1'], 
               dra, 
               alpha=0.5)
    sp = pl.subplot(222)
    sp.scatter(astrometry.comb['dec_1'], 
               dra, 
               alpha=0.5)
    sp = pl.subplot(223)
    sp.scatter(astrometry.comb['ra_1'], 
               ddec, 
               alpha=0.5)
    sp = pl.subplot(224)
    sp.scatter(astrometry.comb['dec_1'], 
               ddec, 
               alpha=0.5)
    
    fig = pl.figure(num=3)
    fig.clf()
    sp = fig.add_subplot(111)
    sp.scatter(astrometry.comb['ra_1'],
               astrometry.comb['dec_1'],
               c=np.hypot(dra, ddec), cmap='jet')
    
    fig = pl.figure(num=2)
    fig.clf()
    sp = fig.add_subplot(111)
    db = DBSCAN(eps=0.15)
    x = np.zeros((len(dra), 2))
    x[:, 0] = dra
    x[:, 1] = ddec
    db.fit(x)
    sp.scatter(dra,
               ddec,
               c=db.labels_)
    p = np.where(db.labels_ >= 0)[0]
    print(round(np.median(dra), 2), round(np.mean(dra), 2), round(np.std(dra), 2))
    print(round(np.median(ddec), 2), round(np.mean(ddec), 2), round(np.std(ddec), 2))
    dra = dra[p]
    ddec = ddec[p]
    print(len(dra))
    print(round(np.median(dra), 2), round(np.mean(dra), 2), round(np.std(dra), 2))
    print(round(np.median(ddec), 2), round(np.mean(ddec), 2), round(np.std(ddec), 2))
    
    apass = Vizier.query_region(s, catalog='APASS', radius=30*u.arcmin)
    apass = apass[-1]
    
    
    fig = pl.figure(num=4)
    fig.clf()
    sp = fig.add_subplot(211)
    
    sources = astrometry.sources
    comb = combine_w_apass(sources, apass)
    
    p = np.where((comb['mag'] > -99) &
                 (comb['Vmag'] > -99) &
                 (comb['e_Vmag'] > 0))[0]
    comb = comb[p]
    
    fit = np.polyfit(comb['mag'], comb['Vmag'], 3, w=1./comb['e_Vmag'])
    poly = np.poly1d(fit)
    
    sp.scatter(comb['mag'], comb['Vmag'])
    comb.sort('mag')
    sp.plot(comb['mag'], poly(comb['mag']), '--k')
    
    sp2 = pl.subplot(212)
    sp2.scatter(comb['Vmag'], comb['Vmag']-poly(comb['mag']))
    print(np.median(np.abs(comb['Vmag']-poly(comb['mag']))))
    
    phot = identify_sources(path+file_name)
    db = DBSCAN(eps=2, min_samples=2)
    x = np.zeros((len(comb)+len(phot), 2))
    x[:len(comb), 0] = comb['xcentroid']
    x[:len(comb), 1] = comb['ycentroid']
    x[len(comb):, 0] = phot['xcentroid']
    x[len(comb):, 1] = phot['ycentroid']
    
    db.fit(x)
    comb['label'] = db.labels_[:len(comb)]
    phot['label'] = db.labels_[len(comb):]
    p = np.where(comb['label'] >= 0)[0]
    comb = comb[p]
    p = np.where(phot['label'] >= 0)[0]
    phot = phot[p]
    
    comb2 = join(comb, phot, keys='label')
    comb2.rename_column('mag_2', 'mag')
    
#    comb2 = combine_w_apass(phot, apass)
    
    sp.scatter(comb2['mag'], comb2['Vmag'])
    fit = np.polyfit(comb2['mag'], comb2['Vmag'], 3, w=1./comb2['e_Vmag'])
    poly = np.poly1d(fit)
    comb2.sort('mag')
    sp.plot(comb2['mag'], poly(comb2['mag']), '--k')
    
    sp2.scatter(comb2['Vmag'], comb2['Vmag']-poly(comb2['mag']))
    sp.scatter(comb2['mag_1'], comb2['mag'])
    
    x1_col = 'ra'
    y1_col = 'dec'
    
    x2_col = 'ra_1'
    y2_col = 'dec_1'
#    x2_col = 'ra_2'
#    y2_col = 'dec_2'
    
    wcs = AWCS(path+file_name)
    print(wcs)
    comb2 = astrometry.comb
    print(comb2.colnames)
    comb2['ra'], comb2['dec'] = wcs.all_pix2world(comb2['xcentroid'],
                                                  comb2['ycentroid'],1)
    dra = (comb2[x1_col]-comb2[x2_col])*3600
    ddec = (comb2[y1_col]-comb2[y2_col])*3600
    print(round(np.median(dra), 2), round(np.mean(dra), 2), round(np.std(dra), 2))
    print(round(np.median(ddec), 2), round(np.mean(ddec), 2), round(np.std(ddec), 2))
    db = DBSCAN(eps=0.15)
    x = np.zeros((len(dra), 2))
    x[:, 0] = dra
    x[:, 1] = ddec
    db.fit(x)
    
    p = np.where(db.labels_ >= 0)[0]
    print(len(p)/len(comb2), len(comb2), len(p))
    comb2 = comb2[p]
    dra = (comb2[x1_col]-comb2[x2_col])*3600
    ddec = (comb2[y1_col]-comb2[y2_col])*3600
    
    pl.clf()
    sp = pl.subplot(221)
    sp.scatter(comb2[x1_col], ddec)
    fit = np.polyfit(comb2[x1_col]-np.median(comb2[x1_col]),
                     (comb2[x1_col]-comb2[x2_col])*3600, 1)
    fit = np.polyfit(comb2[y1_col]-np.median(comb2[y1_col]),
                     ddec, 1)
    dec_rel = comb2[y1_col]-np.median(comb2[y1_col])
    
    dec_cor = np.poly1d(fit)
    print(round(np.median(dra), 2), round(np.mean(dra), 2), round(np.std(dra), 2))
    print(round(np.median(ddec), 2), round(np.mean(ddec), 2), round(np.std(ddec), 2))
    print(fit)
    sp = pl.subplot(222)
    sp.scatter(comb2[y1_col], dra)
    sp.scatter(comb2[y1_col], dec_cor(comb2[y1_col]-np.median(comb2[y1_col])))
    sp = pl.subplot(223)
    sp.scatter(comb2[x1_col],comb2[y1_col], c=ddec, cmap='jet')
    sp.scatter(comb2[x1_col],comb2[y1_col], c=ddec-dec_cor(dec_rel), cmap='jet')
    sp = pl.subplot(224)
    
    sp.scatter(comb2[x1_col],comb2[y1_col], c=dra, cmap='jet')
    
def do_astrometry():
    
    astrometry = Astrometry(path+file_name)
    try:
        astrometry.calibrate()
    except TypeError as e:
        print(e)
        pass
#    print(astrometry.stats)
    
    wcs = AWCS(path+file_name)
    crpix = wcs.wcs.crpix
    delta = (np.array(wcs.all_pix2world(crpix[0],crpix[1], 0))-np.array(wcs.wcs.crval))*3600
    print('wcs test',test_wcs(wcs, astrometry.comb))
    return astrometry

def astronomy_all(path, calib):
    total = []
    gaia = None
    usno = None
    for f in os.listdir(path):
        if '_red' in f:
            astrometry = Astrometry(path+f, gaia=gaia, usno=usno)
            try:
                astrometry.calibrate()
                print(astrometry.sources['jd'][0])
                if gaia is None:
                    gaia = astrometry.gaia
                    usno = astrometry.usno
#                    print(usno.bright)
                sources = astrometry.sources.copy()
                db = DBSCAN(eps=3./3600, min_samples=2)
                x = np.zeros((len(calib)+len(sources), 2))
                x[:len(calib), 0] = calib['ra']
                x[:len(calib), 1] = calib['dec']
                x[len(calib):, 0] = sources['ra']
                x[len(calib):, 1] = sources['dec']
                db.fit(x)
                calib['label'] = db.labels_[:len(calib)]
                sources['label'] = db.labels_[len(calib):]
                p1 = np.where(calib['label'] > -1)[0]
                p2 = np.where(sources['label'] > -1)[0]
                comb = join(calib[['label', 'mag_mean', 'mag_std', 'c']][p1],
                            sources[['label', 'mag']][p2],
                            keys='label')
                fit = np.polyfit(comb['mag'], comb['mag_mean'], 2,
                                 w=comb['c']/comb['mag_std'])
                poly = np.poly1d(fit)
                sources['mag_c'] = poly(sources['mag'])
                total.append(sources)
                
            except TypeError as e:
                print(e)
                pass
    return vstack(total)

def total_analyse(path, calib):
    total = astronomy_all(path, calib)
    
    p = np.where((total['xcentroid'] > 20) &
                 (total['xcentroid'] < 1000) &
                 (total['ycentroid'] > 20) &
                 (total['ycentroid'] < 1000))
    total = total[p]
    db = DBSCAN(eps=3./3600, min_samples=10)
    x = np.zeros((len(total), 2))
    x[:, 0] = total['ra']
    x[:, 1] = total['dec']
    db.fit(x)
    total['label'] = db.labels_
    
    p = np.where(total['label'] > -1)[0]
    total = total[p]
    
    total = total.group_by('label')
    mean = total.groups.aggregate(np.mean)
    std = total.groups.aggregate(np.std)
    std['ra'] *= 3600
    std['dec'] *= 3600
    mean.rename_column('ra', 'ra_mean')
    mean.rename_column('dec', 'dec_mean')
    mean.rename_column('mag_c', 'mag_mean')
    std.rename_column('ra', 'ra_std')
    std.rename_column('dec', 'dec_std')
    std.rename_column('mag_c', 'mag_std')
    stats = join(mean[['label', 'ra_mean', 'dec_mean', 'mag_mean']],
                 std[['label', 'ra_std', 'dec_std', 'mag_std']],
                 keys='label')
    total = join(total, stats, keys='label')
    return total

def plot_stat(total, col1, col2, label1, label2, fig_num):
    fig = pl.figure(num=fig_num)
    fig.clf()
    sp = pl.subplot()
    sp.scatter(total[col1], total[col2], marker='.', c='k')
    sp.set_xlabel(label1)
    sp.set_ylabel(label2)
    
def plot_analyse(total):
    total['dra_jd'] = (total['ra']-total['ra_mean'])*3600
    total['ddec_jd'] = (total['dec']-total['dec_mean'])*3600
    total['dmag_jd'] = (total['mag_c']-total['mag_mean'])
    jds = total.group_by('jd')
    jds = jds.groups.aggregate(np.mean)
    jds['rjd'] = (jds['jd']-jds['jd'].min())*24
    plot_stat(total, 'ra_mean', 'ra_std', 
              'RA [deg]', '$\\sigma_{RA}$ [arcsec]', 1)
    plot_stat(total, 'dec_mean', 'dec_std', 
              'Dec [deg]', '$\\sigma_{Dec}$ [arcsec]', 2)
    plot_stat(total, 'mag_mean', 'mag_std', 
              'mag', '$\\sigma_{mag}$', 3)
    plot_stat(jds, 'rjd', 'dra_jd', 'hours', '$\\Delta$ RA [arcsec]',
              4)
    plot_stat(jds, 'rjd', 'ddec_jd', 'hours', '$\\Delta$ Dec [arcsec]',
              5)
    plot_stat(jds, 'rjd', 'dmag_jd', 'hours', '$\\Delta$ mag',
              6)
if __name__ == '__main__':
#    s = SkyCoord('20h20m0.26s 4d37m55.07s')
    path = '/Users/patrickr/Documents/TEST/2017_10_13/'
    file_name = 'ns142-V-0065_red.fit'
    astrometry =do_astrometry()
#    astrometric_checking()  
#    total = total_analyse(path, calib)
#    plot_analyse(total)