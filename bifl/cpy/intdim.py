from ctypes import CDLL, py_object, RTLD_GLOBAL
import cv
from os.path import abspath, dirname, join

DN = dirname(abspath(__file__))


def intdim(inmat):
    out0 = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_32FC1)
    out1 = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_32FC1)
    out2 = cv.CreateMat(inmat.rows, inmat.cols, cv.CV_32FC1)
    CDLL(join(DN, '_intdim.so'), mode=RTLD_GLOBAL).intdim(
        py_object(inmat),
        py_object(out0),
        py_object(out1),
        py_object(out2),
        )
    return out0, out1, out2


if __name__ == '__main__':
    im = cv.LoadImage('../test.png')
    from colorsplit import colorsplit
    a, _, __ = colorsplit(im)
    a, b, c = intdim(a)
    cv.Normalize(a, a, 0, 255, cv.CV_MINMAX)
    cv.Normalize(b, b, 0, 255, cv.CV_MINMAX)
    cv.Normalize(c, c, 0, 255, cv.CV_MINMAX)
    cv.SaveImage("int0.png", a)
    cv.SaveImage("int1.png", b)
    cv.SaveImage("int2.png", c)
