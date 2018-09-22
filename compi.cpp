#include "mex.h"
#include "class_handle.hpp"


#define OUT_Coef        plhs[0]

void cpd(float *Ptr[6],double *R,double *t,double *N,double *M, double *D,double *sig, double coefficient[640]);
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


    printf("\n \n \n");
    printf("In the compi.cpp MexFunction \n");
    printf("d_X-> %p\n",*(Ptr));
    printf("d_X_C-> %p\n",*(Ptr+1));
    printf("d_Y-> %p\n",*(Ptr+2));
    printf("d_Y_C-> %p\n",*(Ptr+3));
    printf("det_X-> %p\n",*(Ptr+4));
    printf("det_Y-> %p\n",*(Ptr+5));
    double *R,*t,*M,*N,*D,*sig;
    R     = mxGetPr(prhs[6]);
    t     = mxGetPr(prhs[7]);
    N     = mxGetPr(prhs[8]);
    M     = mxGetPr(prhs[9]);    
    D     = mxGetPr(prhs[10]);
    sig   = mxGetPr(prhs[11]);
  
    double coefficient[640];
    cpd(Ptr,R,t,N,M,D,sig,coefficient);
    // Got here, so command not recognized
    int Num;
    if (*D==2)
      Num=45;
    else
      Num=637;

    OUT_Coef     = mxCreateDoubleMatrix(Num, 1, mxREAL);   

    double *Coef;
   
    Coef      = mxGetPr(OUT_Coef);

    //printf("coefficient[0] %f \n",coefficient[0]);
    for (int i=0;i<Num;i++)
    {
     *(Coef+i)=coefficient[i];
     printf("coefficient[%d] %f \n",i,coefficient[i]);
    }


    plhs[1] = convertPtr2Mat<float >(*Ptr);
    plhs[2] = convertPtr2Mat<float >(*(Ptr+1));
    plhs[3] = convertPtr2Mat<float >(*(Ptr+2));
    plhs[4] = convertPtr2Mat<float >(*(Ptr+3));
    plhs[5] = convertPtr2Mat<float >(*(Ptr+4));
    plhs[6] = convertPtr2Mat<float >(*(Ptr+5));
    return;
}
