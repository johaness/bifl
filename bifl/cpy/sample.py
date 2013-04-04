from ctypes import CDLL, py_object
import cv
from os.path import abspath, dirname, join

DN = dirname(abspath(__file__))


def sample(inimage):
    out = cv.CreateMat(inimage.height, inimage.width, cv.CV_32FC1)
    CDLL(join(DN, '_sample.so')).sample(
        py_object(inimage),
        py_object(out),
        )
    return out


if __name__ == '__main__':
    im = cv.LoadImage('../test.png')
    sample(im)
