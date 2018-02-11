from astropy.table import join
from sklearn.cluster import DBSCAN
import numpy as np


def identify_sources(tab1, tab2):
    tab1 = tab1.copy()
    tab2 = tab2.copy()
    x = np.zeros((len(tab1)+len(tab2), 2))
    x[:len(tab1), 0] = tab1['ra']
    x[:len(tab1), 1] = tab1['dec']
    x[len(tab1):, 0] = tab2['ra']
    x[len(tab2):, 1] = tab2['dec']
    db = DBSCAN(eps=5./3600, min_samples=2)
    db.fit(x)
    tab1['label'] = db.labels_[:len(tab1)]
    tab2['label'] = db.labels_[len(tab1):]

    p = np.where(tab1['label'] >= 0)[0]
    tab1 = tab1[p]
    p = np.where(tab2['label'] >= 0)[0]
    tab2 = tab2[p]

    tab = join(tab1, tab2, keys='label')
    return tab


def relative_calibration(current, master):
    """
    Relative calibration to a master file

    :param current: The current photometric data
    :type current: astropy.table.Table
    :param master: The master photometric data
    :type master: astropy.table.Table
    :return: The calibrated input data
    :rtype: astropy.table.Table
    """
    p = np.where(current['fwhm'] <= 1.5*np.median(current['fwhm']))[0]
    current_c = current[p]

    comb = identify_sources(current_c, master)

    rs = np.polyfit(comb['mag_1'], comb['mag_2'], 2, cov=True)
    fit, cov = rs[0], rs[1]

    err = np.sqrt(np.diag(cov))
    poly = np.poly1d(rs[0])

    current['mag_calib'] = poly(current['mag'])
    current.meta.update({'poly_degree': 2,
                         'fit': '{}*x**2+{}*x+{}'.format(*fit),
                         'err': '{}*x**2+{}*x+{}'.format(*err)})
    return current


if __name__ == '__main__':
    from astropy.table import Table
    import pylab as pl
    data = Table.read('../ns142.fits')
    jds = np.unique(data['jd'])
    
    p_s = np.where(data['jd'] == jds[29])[0]
    jd1 = data[p_s]
    p_s = np.where(data['jd'] == jds[30])[0]
    jd2 = data[p_s]
    
    co = join(jd1, jd2, keys='id')
    co.sort('mag_1')
    pl.clf()
    sp = pl.subplot(211)
    sp.scatter(co['mag_1'], co['mag_2'])
    
    fit_mag = np.polyfit(co['mag_1'], co['mag_2'], 2)
    poly_mag = np.poly1d(fit_mag)
    sp.plot(co['mag_1'], poly_mag(co['mag_1']), '--k')
    sp = pl.subplot(212)
    sp.scatter(co['mag_2'], co['mag_2']-poly_mag(co['mag_1']))
