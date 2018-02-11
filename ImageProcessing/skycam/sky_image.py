from multiprocessing import Process
from astropy.table import Table, vstack
from astropy.time import Time
from ImageProcessing.photometry.mag_estimation import source_detection_w_error as sdwe
import imageio
import time
import os
import numpy as np
from datetime import datetime, date
from sklearn.cluster import DBSCAN

import math
from scipy.optimize import curve_fit
import pylab as pl
from matplotlib.colors import LogNorm
from skimage import filters
from scipy.interpolate import interp2d, interp1d
class SkyDetections:
    x_col = 'xcentroid'
    y_col = 'ycentroid'
    x0 = 3096//2
    y0 = 2080//2
    r_max = 1080
    image_time = None
    detections = None
    stats = None

    def __init__(self, detections):
        self.detections = detections
        self.image_time = datetime.now()
        
        try:
            self.__exclude__()
        except ValueError:
            pass

        self.detections['r'] = np.hypot(self.detections[self.x_col]-self.x0,
                                        self.detections[self.y_col]-self.y0)
        p = np.where(self.detections['r'] < self.r_max)[0]
        self.detections = self.detections[p]
        self.detections['phi'] = np.arccos((self.detections[self.x_col]-self.x0)/self.detections['r'])
        p = np.where(self.detections[self.y_col]-self.y0 < 0)[0]
        self.detections['phi'][p] *= -1
#        print(self.detections)
#        self.__make_stats__()

    def __exclude__(self):
        db = DBSCAN(eps=40, min_samples=10)
        p = np.where((self.detections['xcentroid'] > 0) &
                     (self.detections['ycentroid'] > 0))[0]
        print(len(self.detections))
        print(len(p))
        self.detections = self.detections[p]
        x = np.zeros((len(self.detections), 2))
        x[:, 0] = self.detections['xcentroid']
        x[:, 1] = self.detections['ycentroid']
        db.fit(x)
        p = np.where(db.labels_ == -1)[0]
        print(len(p))
        self.detections = self.detections[p]
        
    def __make_stats__(self):
        r_s = np.linspace(0, self.r_max, 8)
        phi_s = np.linspace(-np.pi, np.pi, 8)
        out = []
        for rs, re in zip(r_s[:-1], r_s[1:]):
            p = np.where((self.detections['r'] >= rs) &
                         (self.detections['r'] <= re))[0]
            d = self.detections[p]
            r = (re+rs)/2
            for phis, phie in zip(phi_s[:-1], phi_s[1:]):
                p = np.where((d['phi'] >= phis) &
                             (d['phi'] <= phie))[0]
                # ds = d[p]
                phi = (phie+phis)/2
                out.append((r, phi, len(p)))
        self.stats = Table(rows=out, names=['r', 'phi', 'counts'])
        self.stats['time'] = Time.now().jd
        print(self.stats)


class NightDetection:
    night_date = None
    sky_detections = []

    def __init__(self):
        self.night_date = date.today()

    def add_detections(self, detections):
        dk = SkyDetections(detections)
        print(dk)
        self.sky_detections.append(dk)

    def get_all_stats(self):
        tabs = []
        for s in self.sky_detections:
            tabs.append(s.stats)
        tabs = vstack(tabs)
        return tabs


class SkyImage(Process):
    image = None
    temp_path = ''
    active = True
    detection_tracker = None

    def __init__(self, temp_path='./temp/'):
        Process.__init__(self)
        if temp_path[-1] != '/':
            temp_path += '/'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.temp_path = temp_path
        self.detection_tracker = NightDetection()

    def __read_image__(self):
        self.image = imageio.imread(self.temp_path+'allskycam2.jpg')
        self.image = self.image[:, 510:2850, 1]#np.mean(self.image, axis=-1)
        sobel = filters.sobel(self.image)
        self.image = filters.gaussian(sobel, sigma=2.0)
#        p = np.where(self.image >0)[0]
#        self.image = self.image[p]

    def __process_image__(self):
        detections = sdwe(self.image)
        print(detections)
        self.detection_tracker.add_detections(detections)

    def run(self):
        while self.active:
            self.__read_image__()
            self.__process_image__()
            print('done')
#            time.sleep(60)
            break

def positions_information(image, p):
    image = image[p]
    norm = np.sum(image)
    x = np.sum(image*p[1])/norm
    y = np.sum(image*p[0])/norm
    x2 = np.sum(image*p[1]**2)/norm
    x2 -= x**2
    y2 = np.sum(image*p[0]**2)/norm
    y2 -= y**2
    xy = np.sum(image*p[0]*p[1])/norm
    xy -= x*y
    
    sq = math.sqrt(((x2-y2)/2)**2+xy**2)
    mean = (x2+y2)/2
    a = math.sqrt(mean+sq)
    b = math.sqrt(mean-sq)
    
    theta = np.arctan(2*xy/(x2-y2))/2
    
    cxx = y2/sq
    cyy = x2/sq
    cxy = -2.*xy/sq
    
    return [x, y, x2, y2, xy, a, b, theta, cxx, cyy, cxy]

def extract(image):
    med = np.median(image)
    std = np.nanstd(image[400:800, 400:800])
    print(med, std)
    p = np.where(image >= med+1.5*std)
    print(len(p[0]))
    
    db = DBSCAN(eps=1.2, min_samples=4)
    x = np.zeros((len(p[0]), 2))
    x[:, 0] = p[0]
    x[:, 1] = p[1]
    db.fit(x)
    print(len(np.unique(db.labels_)))
    
    stats = []
    for l in np.unique(db.labels_):
        try:
            pl = np.where(db.labels_ == l)[0]
            pi = (p[0][pl], p[1][pl])
            center = np.mean(x[pl], axis=0)
            total = np.sum(image[pi])
            
            pos = positions_information(image, pi)
            
            row = [l, len(pl), center[0], center[1], total]
            row.extend(pos)
            stats.append(tuple(row))
        except ValueError:
            pass
    stats = Table(rows=stats, names=['labels', 'count', 'y', 'x', 'sum',
                                     'x_c', 'y_c', 'x2', 'y2', 'xy',
                                     'a', 'b', 'theta', 'cxx', 'cyy', 'cxy'])
    p = np.where((stats['count'] < stats['count'].max())&
                 (stats['labels'] >= 0))[0]
    stats = stats[p]
    return stats
    
def reduce_image(image):
    shape = image.shape
    step = 20
    out = []
    for i in range(shape[0]//step):
        row = []
        for j in range(shape[1]//step):
            row.append(np.median(image[i*step:(i+1)*step, j*step:(j+1)*step]))
        out.append(row)
    return np.array(out)
    

def remove_background(image):
    red = reduce_image(image)
    red /= np.nanmean(red)
    x = np.linspace(0, image.shape[1]-1, image.shape[1]/20) 
    y = np.linspace(0, image.shape[0]-1, image.shape[0]/20)
    y0 = np.linspace(0, image.shape[0]-1, image.shape[0])
    x0 = np.linspace(0, image.shape[1]-1, image.shape[1])
    inter = interp2d(x, y, red)
    red = inter(x0, y0)
    img = image/red
    
    img = np.nan_to_num(img)
    p = np.where(img == 0)
    img[p] = np.nanmedian(img)
    p = np.where(img > 0.5)
    img[p] = np.nanmedian(img)
    return img

def rot(x,x0, y0):
    phi = x[len(x)//2:]-y0
    r = x[:len(x)//2]-x0
    
    cosi = np.cos(phi)
    sini = np.sin(phi)
    xn = r*cosi+y0
    yn = r*sini+y0
    return np.append(xn, yn)

def plot_phi_r(stats, fig_num):
    fig = pl.figure(num=fig_num)
    fig.clf()
    sp = pl.subplot()
    r = np.array(stats['r'])
    phi = np.array(stats['phi'])
    r -= np.mean(r)
    r /= np.std(r)
    phi -= np.mean(phi)
    phi /= np.std(phi)
    sp.scatter(phi, r, marker='.')
    
    db = DBSCAN(eps=0.14, min_samples=10)
    x = np.zeros((len(r), 2))
    x[:, 0] = r
    x[:, 1] = phi
    db.fit(x)
    p = np.where(db.labels_ == -1)[0]
    print(len(np.unique(db.labels_)))
    sp.scatter(phi[p], r[p], marker='.')
    p0 = p
    p_ex = []
    
    labels = db.labels_
    for l in np.unique(labels):
        if l == -1:
            continue
        p = np.where(labels == l)[0]
        x = np.append(stats['x'][p], stats['y'][p])
        ra = np.append(stats['r'][p], stats['phi'][p])
        popt, pcov = curve_fit(rot, ra, x)
        if 200 < popt[0] < 2000 or 200 < popt[1] < 2000:
            print(popt)
            print(l)
            print(np.sqrt(np.diag(pcov)))
            p_ex = np.append(p_ex, p)
            continue
        else:
            p0 = np.append(p0, p)
    p_ex = np.array(p_ex, np.int32)
    sp.scatter(phi[p_ex], r[p_ex], marker='.', c=labels[p_ex], cmap='jet')
    
    fig = pl.figure(num=10)
    fig.clf()
    sp = pl.subplot(211)
    sp.hist(stats['r'], bins=100, histtype='step')
    sp = pl.subplot(212)
    sp.scatter(stats['phi'], stats['r'], c=stats['count'],marker='.', 
               cmap='jet', norm=LogNorm())
    
    return stats
    

def reduce_stats(stats, x0, y0):
    print(x0, y0)
    
    p = np.where(stats['count'] < 100)[0]
    stats = stats[p]
    
    xr = (stats['x']-x0)/x0
    yr = (stats['y']-y0)/y0
    stats['r'] = np.hypot(xr,
                          yr)
    stats['phi'] = np.arccos(xr/stats['r'])
    p = np.where(yr < 0)[0]
    stats['phi'][p] *= -1
    
    plot_phi_r(stats, 3)
    p = np.where(stats['r'] < 0.9)[0]
    stats = stats[p]
#    sp.scatter(stats['phi'], stats['r'])
    stats = plot_phi_r(stats, 4)
    
#    p = np.where(stats['sum']/stats['count'] < 0.04)[0]
#    stats = stats[p]
    
#    fit = np.polyfit(stats['count'], stats['sum']/stats['count'], 2)
#    poly = np.poly1d(fit)
#    d_counts = stats['sum']/stats['count']-poly(stats['count'])
#    p = np.where(d_counts > -0.0025)[0]
#    stats = stats[p]
#    p = np.where(stats['count'] < 40)[0]
#    stats = stats[p]
    
    p = np.where(stats['b']/stats['a'] > 0.65)[0]
    stats = stats[p]
    return stats


def plot_process(path):
    image = imageio.imread(path)
    image = image[:, 510:2850, 1]
    sobel = filters.sobel(image)
    sobel = filters.gaussian(sobel, sigma=1.0)
    
    img = remove_background(sobel)
    
#    img_red = reduce_image(image)
    pl.clf()
    fig = pl.figure(num=1)
    fig.clf()
    sp2 = pl.subplot()
    sp2.imshow(img, cmap='gray_r', norm=LogNorm(vmin=np.nanmedian(img),
                                                 ))
    stats = extract(img)
    sp2.scatter(stats['x'], stats['y'], marker='x', c='r')
    stats = reduce_stats(stats, img.shape[1]/2, img.shape[0]/2)
    sp2.scatter(stats['x'], stats['y'], marker='x', c='g')
    fig = pl.figure(num=2)
    fig.clf()
    sp = pl.subplot()
    sp.scatter(stats['labels'], stats['count'])
    
    fig = pl.figure(num=5)
    fig.clf()
    sp = pl.subplot()
    sp.hist2d(stats['x'], stats['y'], bins=20)
    
    fig = pl.figure(num=7)
    fig.clf()
    sp = pl.subplot()
    fit = np.polyfit(stats['count'], stats['sum']/stats['count'], 2)
    poly = np.poly1d(fit)
    d_counts = stats['sum']/stats['count']-poly(stats['count'])
    sp.scatter(stats['count'], d_counts, marker='.')
    stats.sort('count')
    return stats, sobel
    
    
def get_stats(path):
    image = imageio.imread(path)
    image = image[:, 510:2850, 1]
    sobel = filters.sobel(image)
#    sobel = filters.gaussian(sobel, sigma=4.0)
    
    img = remove_background(sobel)
    
#    img_red = reduce_image(image)
    stats = extract(img)
    stats = reduce_stats(stats, img.shape[1]/2, img.shape[0]/2)
    return stats

def show_results(path, fig_num):
    path = path
    stats = get_stats(path)
    fig = pl.figure(num=fig_num)
    fig.clf()
    sp = pl.subplot()
    sp.hist2d(stats['x'], stats['y'], bins=30)
    
if __name__ == '__main__':
    
#    for i in [1,2,3,4,5,6,7,8,9,10]:
#        path = '/Users/patrickr/Downloads/allskycam{}.jpg'.format(i)
#        show_results(path, i)
    stats, img = plot_process('/Users/patrickr/Downloads/allskycam{}.jpg'.format(6))
    
#    si = SkyImage(temp_path='/Users/patrickr/Downloads/')
#    si.run()
        