#!/home/jss/py27/bin/python

import sys
from cPickle import dump

import cv

from log import setup_logging, info
from img import mat2pil
from features import extract
import cvpickle  # noqa


def main():
    setup_logging()
    for fn in sys.argv[1:] or ['input.png']:
        im = cv.LoadImage(fn)
        fts = extract(im)

        pfn = fn + "-features.dat"
        info('Storing feature pickle in %s', pfn)
        dump(fts, file(pfn, 'wb'))

        for l, layer in enumerate(fts):
            for fname, fval in layer.items():
                ffn = '%s-feat-%d-%s.png' % (fn, l, fname,)
                info('Rendering feature %s', ffn)
                mat2pil(fval).save(ffn)


if __name__ == '__main__':
    main()
