"""
Convert OpenCV matrices to PNG using different colormaps
"""

from pkg_resources import resource_stream
import cv
import Image
import numpy as np


jet = np.load(resource_stream(__name__, 'jet.npy')).astype(np.uint8)


def mat2pil(inmat, cmap=jet):
    """
    Normalize inmat to 0..255, apply colormap, return as PIL Image
    """
    lut = jet
    mat1 = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_8UC1)
    cv.Normalize(inmat, mat1, 0, 255, cv.CV_MINMAX)
    mat4 = cv.CreateMat(mat1.rows, mat1.cols, cv.CV_8UC4)
    cv.MixChannels([mat1], [mat4], [(0, 0), (0, 1), (0, 2), (0, 3)])
    out = cv.CreateMat(mat1.rows, mat1.cols, cv.CV_8UC4)
    cv.LUT(mat4, out, cv.fromarray(lut))
    return Image.fromstring('RGBA', cv.GetSize(out), out.tostring())


def mat2gray(inmat, alpha=True):
    """
    Normalize inmat to 0..255, apply grayscale colormap, use inverse
    of input as alpha channel (ie white is transparent), return as PIL Image
    """
    mat1 = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_8UC1)
    cv.Normalize(inmat, mat1, 0, 255, cv.CV_MINMAX)
    im = Image.fromstring('L', cv.GetSize(mat1), mat1.tostring())
    if alpha:
        cv.Not(mat1, mat1)
        al = Image.fromstring('L', cv.GetSize(mat1), mat1.tostring())
        im.putalpha(al)
    return im
