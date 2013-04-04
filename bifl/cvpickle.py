"""
Pickling for cvMats
"""

import cv
import numpy as np
import copy_reg
from zlib import compress, decompress

def cvmat_dump(mat):
    red = list(np.asarray(mat).__reduce__(),)
    rd2 = red[2]
    red = (red[0], red[1],
            (rd2[0], rd2[1], rd2[2], rd2[3], compress(rd2[4]),),)
    return (cvmat_load, red,)

def cvmat_load(c, a, (p0, p1, p2, p3, p4)):
    o = c(*a)
    o.__setstate__((p0, p1, p2, p3, decompress(p4)))
    return cv.fromarray(o)

copy_reg.pickle(cv.cvmat, cvmat_dump, cvmat_load)
