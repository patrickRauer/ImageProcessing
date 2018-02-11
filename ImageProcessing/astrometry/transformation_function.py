from scipy.optimize import curve_fit
import numpy as np
import math


def transform_axis(x, xf, y, second=False):
    if second:
        func = lin_trafo2
    else:
        func = lin_trafo
    fit, cov = curve_fit(func,
                         xf, y)
    n_coordinates = func(x, *fit)
    err = np.sqrt(np.diag(cov))
    return n_coordinates, [fit, err]


def transform_direct(x, x0, y0, ra0, dec0, a, b, c, d):
    y = x[len(x) // 2:] - y0
    x = x[:len(x) // 2] - x0

    xn = a * x + b * y + ra0
    yn = c * x + d * y + dec0
    return np.append(xn, yn)


def transform_direct2(x, x0, y0, ra0, dec0, c, ang):
    y = x[len(x) // 2:] - y0
    x = x[:len(x) // 2] - x0

    sini = math.sin(ang)
    cosi = math.cos(ang)

    xn = cosi * x - sini * y
    yn = sini * x + cosi * y

    xn *= c
    yn *= c

    xn += ra0
    yn += dec0
    return np.append(xn, yn)


def lin_trafo(x, a, b, c):
    y = x[len(x) // 2:]
    x = x[:len(x) // 2]

    n = a * x + b * y + c
    return n


def lin_trafo2(x, a, b):
    y = x[len(x) // 2:]
    x = x[:len(x) // 2]

    xn = a * x - b * y
    return xn
