#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:11:13 2018

@author: patrickr
"""

from mag_estimation import source_detection_w_error
from astropy.io import fits

if __name__ == '__main__':
    path = '/Users/patrickr/Documents/TEST/2017_10_13/'
    file_name = 'ns142-V-0067_red.fit'
    with fits.open(path+file_name) as fi:
        sources = source_detection_w_error(fi[0].data)
    pass