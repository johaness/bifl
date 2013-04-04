from ctypes import CDLL, py_object
import cv
from os.path import dirname, join

DN = dirname(__file__)


def colorsplit(inimage):
    redgreen = cv.CreateMat(inimage.height, inimage.width, cv.CV_32FC1)
    blueyellow = cv.CreateMat(inimage.height, inimage.width, cv.CV_32FC1)
    luminance = cv.CreateMat(inimage.height, inimage.width, cv.CV_32FC1)
    saturation = cv.CreateMat(inimage.height, inimage.width, cv.CV_32FC1)
    CDLL(join(DN, '_colorsplit.so')).colorsplit(
        py_object(inimage),
        py_object(luminance),
        py_object(saturation),
        py_object(redgreen),
        py_object(blueyellow),
        )
    return luminance, saturation, redgreen, blueyellow


