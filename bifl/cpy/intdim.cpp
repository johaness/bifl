#include <opencv2/opencv.hpp>
#include <pycv.hpp>

extern "C" {

#include "intdim_tensor.h"

extern void intdim(
        PyObject * pyInput,
        PyObject * pyOutput0,
        PyObject * pyOutput1,
        PyObject * pyOutput2)
{
    CvMat * in = ((cvmat_t*)pyInput)->a;
    CvMat * out0 = ((cvmat_t *)pyOutput0)->a;
    CvMat * out1 = ((cvmat_t *)pyOutput1)->a;
    CvMat * out2 = ((cvmat_t *)pyOutput2)->a;

    CvMat* t = cvCreateMat(in->rows, in->cols, CV_32FC1);
    CvMat* c = cvCreateMat(in->rows, in->cols, CV_32FC1);
    tensor(in, t, c, 61, 10);

    cvConvertScale(t, out0, -1, 1);
    cvMul(t, c, out1);
    cvSub(t, out1, out2);

    cvReleaseMat(&t);
    cvReleaseMat(&c);
}

}



