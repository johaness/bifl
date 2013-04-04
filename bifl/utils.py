"""
Utils
----

Helper functions for OpenCV <-> NumPy interaction
"""

import cv
import numpy as np


def saveIm(fn, mat):
    cv.Normalize(mat, mat, 0, 255, cv.CV_MINMAX)
    cv.SaveImage(fn, mat)


def sameMat(inmat):
    outmat = cv.CreateMat(inmat.rows, inmat.cols, inmat.type)
    cv.Zero(outmat)
    return outmat


def cv2array(im):
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

    arrdtype=im.depth
    a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width * im.height * im.nChannels)
    a.shape = (im.height, im.width, im.nChannels)
    return a


def array2cv(a):
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
    cv_im = cv.CreateImageHeader((a.shape[1], a.shape[0]),
          dtype2depth[str(a.dtype)], nChannels)
    cv.SetData(cv_im, a.tostring(), a.dtype.itemsize * nChannels * a.shape[1])
    return cv_im


def npsmooth(inar, sigma):
    """smooth for numpy arrays"""
    cvInar = cv.fromarray(inar.astype(np.float32))
    cvOut = sameMat(cvInar)
    cv.Smooth(cvInar, cvOut, cv.CV_GAUSSIAN, 0, 0, sigma)
    out = np.asarray(cvOut)
    return out / out.sum()
