BIFL
====

basic image feature library
---------------------------


Python module with functions for image feature extraction
based on the OpenCV Python bindings.


Requirements
++++++++++++

 * OpenCV (w/ Python bindings: ``sudo apt-get install python-opencv``)
 * Python Image Library


Installation
++++++++++++

``pip install https://github.com/johaness/bifl.git``


Usage
+++++

The ``features.extract`` function allows extraction of all available features
on different spatial scales -- see the source documentation for details.

The ``bifl`` command line script accepts any number of image file names as
parameter, runs ``features.extract`` on every image, stores results in
a pickle file of ``cvMat`` matrices, and renders each feature into a PNG
image.

The other modules contain a number of support functions useful for working
with OpenCV data structures.


Image Features
++++++++++++++

Image features implemented in pure Python are defined in ``mods.py``::

    contrast(inmat, ws = 51)
    smooth(inmat, ws = 51)
    sobel(inmat, ws = 7)

    pyrdown(inmat)
    pyrsdown(*inmats)
    pyrup(inmat)

    zscale(inmat)
    equalize(inmat)

    add(*mats)
    addZ(*mats)
    multiply(inmat, value)
    addZW(inmats, weights)
    addW(inmats, weights)

    spatialbias(inmat, biasmat, (x, y), base=1.0, gain=1.0, bias_zero=None)
    maxior(inmat, steps=10, inhibition=0.2, radius=90)


Image features can be implemented in C with a minimal wrapper below
``bifl/cpy/``::

    # split RGB into RG, BY, Lum, Sat
    colorsplit(inimage)
    # intrinsic dimensionality
    intdim(inmat)


License
+++++++

BSD License, see LICENSE file

Development sponsored by WhiteMatter Labs GmbH, creators of `EyeQuant`_.

.. _EyeQuant: http://eyequant.com
