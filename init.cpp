

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "mex.h"
#include "class_handle.hpp"

void init(
		double* X,
		double* X_C, 
        double* Y,
		double* Y_C,
        double* Coef,
        int N,
		int M,
        int D,
        float *Ptr[6]
        );


/* Input arguments */
#define IN_X		prhs[0]
#define IN_X_cov        prhs[1]
#define IN_Y		prhs[2]
#define IN_Y_cov        prhs[3]



/* Output arguments */
#define OUT_Coef        plhs[0]



/* Gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
  double *X, *Y, *X_C,*Y_C, *Coef,*aa,*bb;
  int     N, M, D;


  /* Get the sizes of each input argument */
  N = mxGetN(IN_X);
  M = mxGetN(IN_Y);
  D = mxGetM(IN_X);
  
  /* Create the new arrays and set the output pointers to them */
  OUT_Coef     = mxCreateDoubleMatrix(1, 1, mxREAL);



    /* Assign pointers to the input arguments */
  X       = mxGetPr(IN_X);
  X_C     = mxGetPr(IN_X_cov);
  Y       = mxGetPr(IN_Y);
  Y_C     = mxGetPr(IN_Y_cov);


  /* Assign pointers to the output arguments */
  Coef      = mxGetPr(OUT_Coef);

  float *Ptr[6];

  /* Do the actual computations in a subroutine */
  init(X, X_C,Y, Y_C,Coef, N, M, D,Ptr);
  printf("\n \n \n");
  printf("Now Go back to MexFunction in init.cpp \n");
  printf("d_X-> %p\n",*(Ptr));
  printf("d_X_C-> %p\n",*(Ptr+1));
  printf("d_Y-> %p\n",*(Ptr+2));
  printf("d_Y_C-> %p\n",*(Ptr+3));
  printf("det_X-> %p\n",*(Ptr+4));
  printf("det_Y-> %p\n",*(Ptr+5));
  

  plhs[1] = convertPtr2Mat<float >(*Ptr);
  plhs[2] = convertPtr2Mat<float >(*(Ptr+1));
  plhs[3] = convertPtr2Mat<float >(*(Ptr+2));
  plhs[4] = convertPtr2Mat<float >(*(Ptr+3));
  plhs[5] = convertPtr2Mat<float >(*(Ptr+4));
  plhs[6] = convertPtr2Mat<float >(*(Ptr+5));
  return;
}

