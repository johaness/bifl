#include <opencv2/opencv.hpp>
#include <pycv.hpp>

using namespace std;

extern "C" {

extern void colorsplit(
        PyObject * pyInput,
        PyObject * pyLuminance,
        PyObject * pySaturation,
        PyObject * pyRedGreen,
        PyObject * pyBlueYellow)
{
    IplImage * in = ((iplimage_t *)pyInput)->a;
    CvMat * out = ((cvmat_t *)pyLuminance)->a;
    CvMat * sat = ((cvmat_t *)pySaturation)->a;
    CvMat * rg = ((cvmat_t *)pyRedGreen)->a;
    CvMat * by = ((cvmat_t *)pyBlueYellow)->a;

    float * LU = out->data.fl;
    float * ST = sat->data.fl;
    float * RG = rg->data.fl;
    float * BY = by->data.fl;
    float lum, norm, bluef;
    float red, green, blue, yellow;
    float r, g, b;
    float vmax, vmin;
    float delta;

    for (int y = 0; y < in->height; y++)
        for (int x = 0; x < in->width; x++)
        {
            r = CV_IMAGE_ELEM(in, uchar, y, x * 3 + 2);
            g = CV_IMAGE_ELEM(in, uchar, y, x * 3 + 1);
            b = CV_IMAGE_ELEM(in, uchar, y, x * 3);
            vmax = max(r, max(g, b));
            vmin = min(r, min(g, b));
            delta = vmax - vmin;
            *ST++ = vmax != 0 ? delta / vmax : 0;

            lum = (r + b + g) / 3.0 / 255.0;
            *LU++ = lum;
            if (lum < 0.03)
            {
                *RG++ = 0;
                *BY++ = 0;
            }
            else
            {
                norm = (1.0 / (r + b + g));
                r *= norm;
                g *= norm;
                b *= norm;
                red = min(1.0, max(0.0, r - 0.5 * (g + b)));
                green = min(1.0, max(0.0, g - 0.5 * (r + b)));
                bluef = b - 0.5 * (r + g);
                yellow = min(1.0, max(0.0, -2.0 * (bluef + fabs(r - g))));
                blue = min(1.0f, max(0.0f, bluef));
                *RG++ = red - green;
                *BY++ = blue - yellow;
            }
        }
}

}



