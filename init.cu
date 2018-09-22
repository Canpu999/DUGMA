
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>


#define MAX_SIZE_DATA 3 
 
 __constant__ int d_Data_Coef[MAX_SIZE_DATA]; //include N,M,D

// copy M,N,D to constant memory
void setup_coef_constant(int N, int M, int D)
{
  const int h_Data_Coef[MAX_SIZE_DATA]={N,M,D};
  // copy data on host to the constant memory on device
  cudaMemcpyToSymbol(d_Data_Coef, h_Data_Coef, MAX_SIZE_DATA * sizeof(int));
}



__global__ void detX_gpu(float *det_X,float *d_X_C)
{
       // calculate the inverse of the covariance and the square root of the determinant of the inverse of the covariance for X
       int N,D;
       N=d_Data_Coef[0];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<N) 
       {
          if (D==2)
          {
            float a1,a2,a3,a4,d;
            a1=*(d_X_C+4*index);
            a2=*(d_X_C+4*index+1);
            a3=*(d_X_C+4*index+2);
            a4=*(d_X_C+4*index+3);
            d=a1*a4-a2*a3;
            if (d>0)
            {
              *(d_X_C+4*index)=a4/d;
              *(d_X_C+4*index+1)=-1*a2/d;
              *(d_X_C+4*index+2)=-1*a3/d;
              *(d_X_C+4*index+3)=a1/d;
              *(det_X+index)=sqrtf(1/d);

            }
            else
            {
              printf("the determint of covariance of %d point in X point cloud is below zero \n", index+1);
              //neglect this point
              *(d_X_C+4*index)=100000;
              *(d_X_C+4*index+1)=100000;
              *(d_X_C+4*index+2)=100000;
              *(d_X_C+4*index+3)=100000;
              *(det_X+index)=0;
            }
          }
          else
          {
            // 3 dimension
    	        float a11,a12,a13,a21,a22,a23,a31,a32,a33,d;
      	        a11=*(d_X_C+9*index);
     	        a21=*(d_X_C+9*index+1);
          	a31=*(d_X_C+9*index+2);
		a12=*(d_X_C+9*index+3);
                a22=*(d_X_C+9*index+4);
                a32=*(d_X_C+9*index+5);
                a13=*(d_X_C+9*index+6);
                a23=*(d_X_C+9*index+7);
                a33=*(d_X_C+9*index+8);
            	d=a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31;
            	if (d>0)
            	{
              	  *(d_X_C+9*index)=(a22*a33 - a23*a32)/d;
                  *(d_X_C+9*index+1)=-1*(a21*a33 - a23*a31)/d;
                  *(d_X_C+9*index+2)=(a21*a32 - a22*a31)/d;
                  *(d_X_C+9*index+3)=-1*(a12*a33 - a13*a32)/d;
                  *(d_X_C+9*index+4)=(a11*a33 - a13*a31)/d;
                  *(d_X_C+9*index+5)=-1*(a11*a32 - a12*a31)/d;
                  *(d_X_C+9*index+6)=(a12*a23 - a13*a22)/d;
                  *(d_X_C+9*index+7)=-1*(a11*a23 - a13*a21)/d;
                  *(d_X_C+9*index+8)=(a11*a22 - a12*a21)/d;
                  *(det_X+index)=sqrtf(1/d);
                }
             else
                {
            	  printf("the determint of covariance of %d point in X point cloud is below zero \n", index+1);
              	  //neglect this point
              	  *(d_X_C+9*index)=100000;
                  *(d_X_C+9*index+1)=100000;
                  *(d_X_C+9*index+2)=100000;
                  *(d_X_C+9*index+3)=100000;
              	  *(d_X_C+9*index+4)=100000;
                  *(d_X_C+9*index+5)=100000;
                  *(d_X_C+9*index+6)=100000;
                  *(d_X_C+9*index+7)=100000;
                  *(d_X_C+9*index+8)=100000;
                  *(det_X+index)=0;
                }
          }
       }
       //printf("double to float: X \n");
     
}

__global__ void detY_gpu(float *det_Y,float *d_Y_C)
{
       // calculate the inverse of the covariance and the square root of the determinant of the inverse of the covariance 
       int M,D;
       M=d_Data_Coef[1];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<M) 
       {
          if (D==2)
          {
            float a1,a2,a3,a4,d;
            a1=*(d_Y_C+4*index);
            a2=*(d_Y_C+4*index+1);
            a3=*(d_Y_C+4*index+2);
            a4=*(d_Y_C+4*index+3);
            d=a1*a4-a2*a3;
            if (d>0)
            {
              *(d_Y_C+4*index)=a4/d;
              *(d_Y_C+4*index+1)=-1*a2/d;
              *(d_Y_C+4*index+2)=-1*a3/d;
              *(d_Y_C+4*index+3)=a1/d;
              *(det_Y+index)=sqrtf(1/d);
         
            }
            else
            {
              printf("the determinant of covariance of %d point in Y point Cloud is below zero \n", index+1);
              //neglect this point
              *(d_Y_C+4*index)=100000;
              *(d_Y_C+4*index+1)=100000;
              *(d_Y_C+4*index+2)=100000;
              *(d_Y_C+4*index+3)=100000;
              *(det_Y+index)=0;  //neglect this point
            }
          }
          else
          {
            // 3 dimension
    	        float a11,a12,a13,a21,a22,a23,a31,a32,a33,d;
      	        a11=*(d_Y_C+9*index);
     	        a21=*(d_Y_C+9*index+1);
          	a31=*(d_Y_C+9*index+2);
		a12=*(d_Y_C+9*index+3);
                a22=*(d_Y_C+9*index+4);
                a32=*(d_Y_C+9*index+5);
                a13=*(d_Y_C+9*index+6);
                a23=*(d_Y_C+9*index+7);
                a33=*(d_Y_C+9*index+8);
            	d=a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31;
            	if (d>0)
            	{
              	  *(d_Y_C+9*index)=(a22*a33 - a23*a32)/d;
                  *(d_Y_C+9*index+1)=-1*(a21*a33 - a23*a31)/d;
                  *(d_Y_C+9*index+2)=(a21*a32 - a22*a31)/d;
                  *(d_Y_C+9*index+3)=-1*(a12*a33 - a13*a32)/d;
                  *(d_Y_C+9*index+4)=(a11*a33 - a13*a31)/d;
                  *(d_Y_C+9*index+5)=-1*(a11*a32 - a12*a31)/d;
                  *(d_Y_C+9*index+6)=(a12*a23 - a13*a22)/d;
                  *(d_Y_C+9*index+7)=-1*(a11*a23 - a13*a21)/d;
                  *(d_Y_C+9*index+8)=(a11*a22 - a12*a21)/d;
                  *(det_Y+index)=sqrtf(1/d);
                }
             else
                {
            	  printf("the determint of covariance of %d point in X point cloud is below zero \n", index+1);
              	  //neglect this point
              	  *(d_Y_C+9*index)=100000;
                  *(d_Y_C+9*index+1)=100000;
                  *(d_Y_C+9*index+2)=100000;
                  *(d_Y_C+9*index+3)=100000;
              	  *(d_Y_C+9*index+4)=100000;
                  *(d_Y_C+9*index+5)=100000;
                  *(d_Y_C+9*index+6)=100000;
                  *(d_Y_C+9*index+7)=100000;
                  *(d_Y_C+9*index+8)=100000;
                  *(det_Y+index)=0;
                }
          }
       }
       //printf("double to float: X \n");
     
}

__global__ void d2f_X_gpu(double *td_X,float *d_X)
{
       int N,D;
       N=d_Data_Coef[0];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<N*D) *(d_X+index)=__double2float_rn(*(td_X+index));
       //printf("double to float: X \n");
     
}

__global__ void d2f_X_C_gpu(double *td_X_C,float *d_X_C)
{
       int N,D;
       N=d_Data_Coef[0];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<N*D*D)  *(d_X_C+index)=__double2float_rn(*(td_X_C+index));
       //printf("double to float: X_C \n");
      
}

__global__ void d2f_Y_gpu(double *td_Y,float *d_Y)
{
       int M,D;
       M=d_Data_Coef[1];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<M*D) *(d_Y+index)=__double2float_rn(*(td_Y+index));
       //printf("double to float: Y \n");
       
}

__global__ void d2f_Y_C_gpu(double *td_Y_C,float *d_Y_C)
{
       int M,D;
       M=d_Data_Coef[1];
       D=d_Data_Coef[2];
       int index=threadIdx.x+blockIdx.x*blockDim.x;
       if (index<M*D*D) *(d_Y_C+index)=__double2float_rn(*(td_Y_C+index));
       //printf("double to float: Y_C \n");
     
}


void check(cudaError_t error_id)
{
  if (error_id != cudaSuccess) {
  printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id,cudaGetErrorString(error_id));
  printf("Result = FAIL\n");
  exit(EXIT_FAILURE);
  }
}

// Ptr[6] is the device global memory address for d_X,d_X_C,d_Y,d_Y_C,det_X,det_Y
// Pt_c is the device constant memory address for M,N,D

void init(
		double* h_X,
		double* h_X_C, 
        double* h_Y,
		double* h_Y_C,
        double* Coef,
        int N,
		int M,
        int D,
        float *Ptr[6]
        )

{ 


  int nBytes_X = N * D * sizeof(float);
  int nBytes_X_C = N * (D*D) * sizeof(float);
  int nBytes_Y = M * D * sizeof(float);
  int nBytes_Y_C = M * (D*D) * sizeof(float);


  // malloc device global memory 
  float *d_X, *d_X_C, *d_Y,*d_Y_C;
  
  check(cudaMalloc((void **)&d_X, nBytes_X));
  check(cudaMalloc((void **)&d_X_C, nBytes_X_C));
  check(cudaMalloc((void **)&d_Y, nBytes_Y));
  check(cudaMalloc((void **)&d_Y_C, nBytes_Y_C));

  // malloc temporal decive global memory
  // in order to convert the double type to the float type, double 8 bytes, float 4 bytes, so there is a coefficient 2
  double *td_X, *td_X_C, *td_Y,*td_Y_C;
  check(cudaMalloc((void **)&td_X, 2*nBytes_X));
  check(cudaMalloc((void **)&td_X_C, 2*nBytes_X_C));
  check(cudaMalloc((void **)&td_Y, 2*nBytes_Y));
  check(cudaMalloc((void **)&td_Y_C, 2*nBytes_Y_C));


  // transfer data from host to temporal place on device
  check(cudaMemcpy(td_X, h_X, 2*nBytes_X, cudaMemcpyHostToDevice));
  check(cudaMemcpy(td_X_C, h_X_C, 2*nBytes_X_C, cudaMemcpyHostToDevice));
  check(cudaMemcpy(td_Y, h_Y, 2*nBytes_Y, cudaMemcpyHostToDevice));
  check(cudaMemcpy(td_Y_C, h_Y_C, 2*nBytes_Y_C, cudaMemcpyHostToDevice));

  check(cudaDeviceSynchronize());

  // setup the constants in the constant memory on device
  setup_coef_constant(N, M, D);
  
  // covert double to float
  int block=1024;
  d2f_X_gpu<<<(D*N+block-1)/block,block>>>(td_X,d_X);
  d2f_X_C_gpu<<<(D*D*N+block-1)/block,block>>>(td_X_C,d_X_C);
  d2f_Y_gpu<<<(D*M+block-1)/block,block>>>(td_Y,d_Y);
  d2f_Y_C_gpu<<<(D*D*M+block-1)/block,block>>>(td_Y_C,d_Y_C);
  
  check(cudaDeviceSynchronize());

  float * det_X, *det_Y; 
  check(cudaMalloc((void **)&det_X, nBytes_X/D));
  check(cudaMalloc((void **)&det_Y, nBytes_Y/D));

  // Calculate the square root for the determint of the inverse covariance
  block=1024;
  detX_gpu<<<(N+block-1)/block,block>>>(det_X,d_X_C);
  detY_gpu<<<(M+block-1)/block,block>>>(det_Y,d_Y_C);
  
  check(cudaDeviceSynchronize());


  // free temporal device global memory
  check(cudaFree(td_X));
  check(cudaFree(td_X_C));
  check(cudaFree(td_Y));
  check(cudaFree(td_Y_C));
  
  // return the device memory address
  Ptr[0]=d_X;
  Ptr[1]=d_X_C;   
  Ptr[2]=d_Y;
  Ptr[3]=d_Y_C;
  Ptr[4]=det_X;   
  Ptr[5]=det_Y;

  printf("Now I am in the init.cu \n");
  printf("d_X-> %p\n",Ptr[0]);
  printf("d_X_C-> %p\n",Ptr[1]);
  printf("d_Y-> %p\n",Ptr[2]);
  printf("d_Y_C-> %p\n",Ptr[3]);
  printf("det_X-> %p\n",Ptr[4]);
  printf("det_Y-> %p\n",Ptr[5]);
 


  // if it is successful, Coef should be 1 
  double c;
  c=1;
  *Coef=c;
  return;
}
