"""
Extract feature batteries from gauss pyramids
"""

import cv
from utils import saveIm
from mods import *
from cpy import *


def stage(lum, sat, rg, by):
    lumc = contrast(lum)
    lumt = contrast(lumc, 251)
    sats = smooth(sat)
    satc = contrast(sat)
    satt = contrast(satc, 251)
    rgc = contrast(rg)
    rgt = contrast(rgc, 251)
    byc = contrast(by)
    byt = contrast(byc, 251)
    sob = sobel(lum)
    sobs = smooth(sob)
    lums = smooth(lum)
    rgs = smooth(rg)
    bys = smooth(by)
    id0, id1, id2 = intdim(lum)
    idX = add(zscale(id0), zscale(id2))
    return dict(lumc=lumc, lumt=lumt, satc=satc, satt=satt, rgc=rgc, rgt=rgt,
            byc=byc, byt=byt, sobs=sobs, lums=lums, id0=id0, id1=id1, id2=id2,
            rgs=rgs, sats=sats, bys=bys, idX=idX,)


def noscale(indict):
    return indict


def zscaledict(indict):
    return dict((n, zscale(m)) for n, m in indict.items())


def histeqdict(indict):
    def eq(inmat):
        m = zscale(inmat)
        return equalize(m)
    return dict((n, eq(m)) for n, m in indict.items())


def pyramid(lsrb, count=3, scaler=noscale):
    """
    run stage in a downwards pyramid for ``count`` times,
    scale each map with ``scaler``,
    return list with one dict per pyramid level
    """
    features = [scaler(stage(*lsrb))]
    if count == 1:
        return features
    lsrb = list(pyrsdown(*lsrb))
    features += pyramid(lsrb, count - 1, scaler)
    return features


def base(im, layers):
    """make sure im's dimensions are multiples of 2**layers"""
    mod = 2 ** layers
    if im.width % mod != 0 or im.height % mod != 0:
        im = cv.GetSubRect(im, (
            0, 0,
            im.width - im.width % mod,
            im.height - im.height % mod,))
    return cv.GetImage(im)


def extract(image, pyr_levels=3, scaler=zscaledict):
    """extract features from ``image``"""
    image = base(image, pyr_levels)
    lsrb = colorsplit(image)
    return pyramid(lsrb, pyr_levels, scaler=scaler)
