#include "mex.h"
#include "class_handle.hpp"


#define OUT_Coef        plhs[0]

void fr_m(float *Ptr[6]);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	 
    // Get the class instance pointer from the second input
    
    float  *d_X = convertMat2Ptr<float >(prhs[0]);
    float  *d_X_C = convertMat2Ptr<float >(prhs[1]);
    float  *d_Y = convertMat2Ptr<float >(prhs[2]);
    float  *d_Y_C = convertMat2Ptr<float >(prhs[3]);
    float  *det_X = convertMat2Ptr<float >(prhs[4]);
    float  *det_Y = convertMat2Ptr<float >(prhs[5]);
    float *Ptr[6];
    Ptr[0]=d_X;
    Ptr[1]=d_X_C;   
    Ptr[2]=d_Y;
    Ptr[3]=d_Y_C;
    Ptr[4]=det_X;   
    Ptr[5]=det_Y;

    fr_m(Ptr);

    // Got here, so command not recognized
    OUT_Coef     = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *Coef;
    Coef      = mxGetPr(OUT_Coef);
    *Coef=1;

    return;
}
