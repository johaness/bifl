/*
 * structs copied from opencv2 Python module cv2.cv.hpp
 */

#include <Python.h>

struct cvmat_t {
  PyObject_HEAD
  CvMat *a;
  PyObject *data;
  size_t offset;
};

struct iplimage_t {
  PyObject_HEAD
  IplImage *a;
  PyObject *data;
  size_t offset;
};
