/*
 * Original Matlab implementation by Hannes Saal
 * Initial C port by Niklas Wilming, edits by Johannes Steger
 */
void tensor(CvMat* img,CvMat* T, CvMat* C, int kernel_size, double std)
{
    CvMat *D1, *D2;
    // Filters for image derivative in x and y direction.
    D1  = cvCreateMat(3,1,CV_32FC1);
    CV_MAT_ELEM(*D1,float,0,0) = -0.5;
    CV_MAT_ELEM(*D1,float,1,0) = -0;
    CV_MAT_ELEM(*D1,float,2,0) =  0.5;
    D2  = cvCreateMat(1,3,CV_32FC1);
    CV_MAT_ELEM(*D2,float,0,0) = -0.5;
    CV_MAT_ELEM(*D2,float,0,1) = -0;
    CV_MAT_ELEM(*D2,float,0,2) =  0.5;

    //  Some temporary matrices
    CvMat* Dx, *Dy, *Dxy, *M;
    double atanh_08 = 10.986122886681097821082175869378261268138885498046875 ;

    // Convolve image with kernel to get image derivatives
    Dx = cvCreateMat(img->rows, img->cols, CV_32FC1);
    Dy = cvCreateMat(img->rows, img->cols, CV_32FC1);

    cvFilter2D(img, Dx, D1);
    cvFilter2D(img, Dy, D2);

    //  Take result to power of2
    Dxy = cvCreateMat(img->rows, img->cols, CV_32FC1);
    cvPow(Dx, Dx, 2);
    cvPow(Dy, Dy, 2);
    cvMul(Dx, Dy, Dxy);

    // Convolve with gaussian
    cvSmooth(Dx, Dx, CV_GAUSSIAN, kernel_size, kernel_size);
    cvSmooth(Dy, Dy, CV_GAUSSIAN, kernel_size, kernel_size);
    cvSmooth(Dxy,Dxy,CV_GAUSSIAN, kernel_size, kernel_size);

    // calculate 0D  vs 1/2D measure
    cvAdd(Dx, Dy, T);

    // Normalize such that in [0,1]
    M = cvCreateMat(img->rows, img->cols, CV_32FC1);
    cvSmooth(T, M, CV_GAUSSIAN, 81, 81, 20);
    float mean  = cvAvg(T).val[0];

    for (int r = 0;  r < T->rows; r++)
        for (int c = 0; c < T->cols; c++)
        {
            float m = CV_MAT_ELEM(*M, float, r, c);
            float t = CV_MAT_ELEM(*T, float, r, c);
            float res = tanh( (atanh_08  / (9*m + mean)) * t );
            CV_MAT_ELEM(*T, float, r, c) = cvIsNaN(res) ? 0 : res;
        }
    cvPow(T, T, 3);
    // Calculate Coherence
    // note: Dy == Ax2 and Dx == Ay2
    cvAdd(Dx, Dy, M);  // M  = Dx + Dy
    cvSub(Dx, Dy,Dx); // Dx = Dx - Dy
    cvPow(Dx, Dx, 2); // Dx = (Dx - Dy) ** 2
    cvPow(Dxy,Dxy,2); // Dxy = Dxy ** 2
    cvConvertScale(Dxy, Dxy, 4, 0); // Dxy = 4 * (Dxy ** 2)
    cvAdd(Dx, Dxy,Dx); // Dx = (Dx - Dy) ** 2 + 4 * (Dxy ** 2)
    cvPow(Dx, Dx, 0.5); // Dx = sqrt( (Di - Dy) ** 2 + 4 * (Dxy ** 2) )
    cvDiv(Dx, M,  C); // C = sqrt( (Dx - Dy) ** 2 + 4 * (Dxy ** 2) ) / (Dx + Dy)

    // zero out nans
    for (int r = 0;  r < C->rows; r++)
        for (int c = 0; c < C->cols; c++)
            if (cvIsNaN(CV_MAT_ELEM(*C, float, r, c)))
                CV_MAT_ELEM(*C, float, r, c) = 0;
    // Release Matrices
    cvReleaseMat(&Dx);
    cvReleaseMat(&Dy);
    cvReleaseMat(&Dxy);
    cvReleaseMat(&M);
}

