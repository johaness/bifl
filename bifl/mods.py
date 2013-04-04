"""
Feature Functions
"""

import cv
import numpy as np

from utils import sameMat


class ZeroFeatureException(ZeroDivisionError):
    pass


def contrast(inmat, ws = 51):
    sq = sameMat(inmat)
    smth = sameMat(inmat)
    outmat = sameMat(inmat)
    cv.Mul(inmat, inmat, sq)
    cv.Smooth(sq, sq, cv.CV_GAUSSIAN, ws, ws)
    cv.Smooth(inmat, smth, cv.CV_GAUSSIAN, ws, ws)
    cv.Mul(smth, smth, smth)
    cv.Sub(sq, smth, outmat)
    cv.Normalize(outmat, outmat, 0, 1, cv.CV_MINMAX)
    cv.Pow(outmat, outmat, 0.5)
    cv.Threshold(outmat, outmat, 0, 0, cv.CV_THRESH_TOZERO)
    del smth, sq
    return outmat


def smooth(inmat, ws = 51):
    outmat = sameMat(inmat)
    cv.Smooth(inmat, outmat, cv.CV_GAUSSIAN, ws, ws)
    return outmat


def sobel(inmat, ws = 7):
    h = sameMat(inmat)
    v = sameMat(inmat)
    outmat = sameMat(inmat)
    cv.Sobel(inmat, h, 1, 0, ws)
    cv.Sobel(inmat, v, 0, 1, ws)
    cv.Mul(h, h, h)
    cv.Mul(v, v, v)
    cv.Add(h, v, outmat)
    cv.Pow(outmat, outmat, 0.5)
    return outmat


def pyrdown(inmat):
    outmat = cv.CreateMat(inmat.rows / 2, inmat.cols / 2, cv.CV_32FC1)
    cv.PyrDown(inmat, outmat)
    return outmat


def pyrsdown(*inmats):
    for i in inmats:
        yield pyrdown(i)


def pyrup(inmat):
    outmat = cv.CreateMat(inmat.rows * 2, inmat.cols * 2, cv.CV_32FC1)
    cv.PyrUp(inmat, outmat)
    return outmat


def zscale(inmat):
    outmat = cv.CloneMat(inmat)
    mean, dev = cv.AvgSdv(outmat)
    if dev[0] != 0:
        cv.SubS(inmat, mean, outmat)
        cv.Scale(outmat, outmat, 1.0 / dev[0])
    else:
        raise ZeroFeatureException("Zero mat %r" % (inmat,))
    return outmat


def multiply(inmat, value):
    outmat = sameMat(inmat)
    cv.Scale(inmat, outmat, value)
    return outmat


def add(*mats):
    outmat = cv.CloneMat(mats[0])
    for mat in mats[1:]:
        cv.Add(outmat, mat, outmat)
    return outmat


def addZ(*mats):
    outmat = zscale(cv.CloneMat(mats[0]))
    for mat in mats[1:]:
        cv.Add(outmat, zscale(mat), outmat)
        #cv.Add(outmat, equalize(zscale(mat)), outmat)
    return outmat


def addZW(inmats, weights):
    mats = inmats.items()
    outmat = sameMat(mats[0][1])
    for key, mat in mats:
        tmp = sameMat(mat)
        cv.Scale(zscale(mat), tmp, weights[key])
        cv.Add(outmat, tmp, outmat)
    return outmat


def addW(inmats, weights):
    mats = inmats.items()
    outmat = sameMat(mats[0][1])
    for key, mat in mats:
        if not key in weights:
            continue
        tmp = sameMat(mat)
        cv.Scale(mat, tmp, weights[key])
        cv.Add(outmat, tmp, outmat)
    return outmat


def equalize(inmat):
    mat = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_8UC1)
    cv.Normalize(inmat, inmat, 0, 255, cv.CV_MINMAX)
    cv.Convert(inmat, mat)
    cv.EqualizeHist(mat, mat)
    return mat


def spatialbias(inmat, biasmat, (x, y), base=1.0, gain=1.0, bias_zero=None):

    outmat = sameMat(inmat)
    if bias_zero is not None:
        ZPX, ZPY = bias_zero
    else:
        ZPX, ZPY = biasmat.cols / 2, biasmat.rows / 2

    # extract bias source subrect
    srcX = max(ZPX - x, 0)
    srcY = max(ZPY - y, 0)
    srcW = min(min(ZPX - x, 0) + inmat.cols, biasmat.cols - srcX)
    srcH = min(min(ZPY - y, 0) + inmat.rows, biasmat.rows - srcY)
    srcRect = cv.GetSubRect(biasmat, (srcX, srcY, srcW, srcH))

    # bias in size and coordinates of target matrix
    tarbias = sameMat(inmat)
    cv.Set(tarbias, 0)

    # copy bias subrect to full size bias
    tarX = max(x - ZPX, 0)
    tarY = max(y - ZPY, 0)

    tarRect = cv.GetSubRect(tarbias, (tarX, tarY, srcW, srcH))
    cv.Copy(srcRect, tarRect)

    cv.Normalize(tarbias, tarbias, base, base + gain, cv.CV_MINMAX)
    cv.Normalize(inmat, inmat, 0, 1, cv.CV_MINMAX)
    cv.Mul(inmat, tarbias, outmat)
    return outmat


def maxior(inmat, steps=10, inhibition=0.2, radius=90):
    sal = cv.CloneMat(inmat)
    mul = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_32FC1)
    cvg = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_32FC1)
    cv.Zero(cvg)
    walk = []
    for step in range(steps):
        iv, xv, il, xl = cv.MinMaxLoc(sal)
        walk.append(xl)
        cv.Set(mul, 1.0)
        cv.Circle(mul, xl, radius, inhibition, -1)
        cv.Circle(cvg, xl, radius, 255, -1)
        cv.Mul(mul, sal, sal)
    cv.Smooth(cvg, cvg, cv.CV_GAUSSIAN, 51, 51)
    return np.array(walk), cvg
