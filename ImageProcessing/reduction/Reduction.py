from astropy.io import fits
from astropy.table import Table
import numpy as np
from datetime import datetime
import os


def collect_meta_file_data(path):
    """
    Collects meta information from the images in the diretory

    :param path: Path to the directory
    :type path: str
    :return: A Table with the file names, the image types, the object names and the exposure time
    :rtype: astropy.table.Table
    """
    if path[-1] != '/':
        path += '/'

    out = []
    for f in os.listdir(path):
        try:
            with fits.open(path+f) as fi:
                header = fi[0].header
                if 'MASTER' in header:
                    continue
                out.append((f, header['IMAGETYP'], header['OBJECT'], header['EXPTIME'], header['FILTER']))
        except IOError:
            pass
    out = Table(rows=out, names=['file_name', 'type', 'object', 'exposure_time', 'filter'])
    return out


def add_header_items(header, **kwargs):
    for keys in kwargs:
        header[keys] = tuple(kwargs[keys])
    return header


def median_image(meta, path):
    """
    Creates a median image from the images in the meta table.

    :param meta: Table with the meta information of the images
    :type meta: astropy.table.Table
    :param path: The path to the directory with images
    :type path: str
    :return: The median of the images and the number of images for the median
    :rtype: numpy.ndarray, int
    """
    out = []
    for m in meta:
        with fits.open(path+m['file_name']) as fi:
            out.append(fi[0].data)
    return np.median(out, axis=0), len(out)


def create_master_darks(meta, path):
    p = np.where(meta['type'] == 'dark')[0]
    darks = meta[p]

    for exp in np.unique(darks['exposure_time']):
        p = np.where(darks['exposure_time'] == exp)[0]
        darks_0 = darks[p]

        dark_images, l = median_image(darks_0, path)
        hdu = fits.PrimaryHDU(data=dark_images)
        hdu.header = add_header_items(hdu.header,
                                      EXPTIME=(exp, 'Exposure time'),
                                      IMGNO=(l, 'Number of dark images'),
                                      CREADATE=(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), 'Time of creation'),
                                      MASTER=(True, 'True if it is a master file, else False'))
        hdu.writeto('{}master_dark_{}.fits'.format(path, exp))


def create_master_flats(meta, path):
    p = np.where(meta['type'] == 'flat')[0]
    darks = meta[p]

    for filt in np.unique(darks['filter']):
        p = np.where(darks['filter'] == filt)[0]
        darks_0 = darks[p]

        dark_images, l = median_image(darks_0, path)
        dark_images = np.float32(dark_images)

        with fits.open('{}master_dark_{}.fits'.format(path, darks_0['exposure_time'][0])) as fi:
            dark = fi[0].data
            dark_images -= dark
            dark_images /= np.mean(dark_images)
        hdu = fits.PrimaryHDU(data=dark_images)
        hdu.header = add_header_items(hdu.header,
                                      FILTER=(filt, 'Filter of the image'),
                                      IMGNO=(l, 'Number of dark images'),
                                      CREADATE=(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), 'Time of creation'),
                                      MASTER=(True, 'True if it is a master file, else False'))
        hdu.writeto('{}master_flat_{}.fits'.format(path, filt))


def reduce_image(path, file_name):
    """
    Apply a dark and flat field correction

    :param path: Path to the directory with images
    :type path: str
    :param file_name: File name of the image
    :type file_name: str
    :return:
    """
    with fits.open(path+file_name) as fi:
        d = fi[0].data

        flat_path = '{}master_flat_{}.fits'.format(path, fi[0].header['FILTER'])
        with fits.open(flat_path) as fi_f:
            flat = fi_f[0].data

        dark_path = '{}master_dark_{}.fits'.format(path, int(fi[0].header['EXPTIME']))
        with fits.open(dark_path) as fi_d:
            dark = fi_d[0].data

        d = np.float32(d)
        d -= dark
        d /= flat

        hdu = fits.PrimaryHDU(data=d, header=fi[0].header)
        hdu.header = add_header_items(hdu.header,
                                      DARKNAME=(dark_path, 'Path to the master dark'),
                                      FLATNAME=(flat_path, 'Path to the master flat'),
                                      CREADATE=(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), 'Time of creation'))
        hdu.writeto(path+file_name.split('.fit')[0]+'_red.fit', overwrite=True)


def create_master_files(path):
    meta = collect_meta_file_data(path)
    create_master_darks(meta, path)
    create_master_flats(meta, path)
    
if __name__ == '__main__':
    path = '/Users/patrickr/Documents/TEST/2017_10_13/'
    for f in os.listdir(path):
        if 'dark' not in f and 'flat' not in f and '_red' not in f:
            reduce_image(path, f)
