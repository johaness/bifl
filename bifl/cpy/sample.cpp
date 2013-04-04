#include <opencv2/opencv.hpp>
#include <pycv.hpp>

extern "C" {

extern void sample(
        PyObject * pyInput,
        PyObject * pyOutput)
{
    IplImage * input = ((iplimage_t *)pyInput)->a;
    CvMat * out = ((cvmat_t *)pyOutput)->a;
    std::cout << "Img " << input->width << " "
      << input->height << std::endl;
    std::cout << "CvMat " << out->rows << " "
      << out->cols << std::endl;
}

}



