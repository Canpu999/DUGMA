
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>



#define MAX_SIZE_DATA 3 
#define MAX_SIZE_TRAN 12
#define times2D 1      // the coefficient matrix in shared memory should be 46*32*times2D
#define times3D 1      // the coefficient matrix in shared memory should be 4*637*times3D   
 
 __constant__ int d_Data_Coef[MAX_SIZE_DATA]; //include N,M,D
 __constant__ double d_Tran_Coef[MAX_SIZE_TRAN];
 __constant__ double d_Sig_Coef[1];

void check(cudaError_t error_id)
{
  if (error_id != cudaSuccess) {
  printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id,cudaGetErrorString(error_id));
  printf("Result = FAIL\n");
  exit(EXIT_FAILURE);
  }
}


// copy M,N,D,R,t to constant memory
void setup_coef_constant(int N, int M, int D, double sig, double *R, double *t)
{
  const int h_Data_Coef[MAX_SIZE_DATA]={N,M,D};
  const double h_Sig_Coef[1]={sig};
  // copy data on host to the constant memory on device
  check(cudaMemcpyToSymbol(d_Sig_Coef, h_Sig_Coef, 1 * sizeof(double)));
  check(cudaMemcpyToSymbol(d_Data_Coef, h_Data_Coef, MAX_SIZE_DATA * sizeof(int)));

  check(cudaDeviceSynchronize());

  double h_Tran_Coef[MAX_SIZE_TRAN];
  if (D==2)
  {
    for (int i=0;i<4;i++)
    {
       h_Tran_Coef[i]=*(R+i);
    }   
    for (int i=0;i<2;i++)
    {
       h_Tran_Coef[i+4]=*(t+i);
    }   
    for (int i=0;i<6;i++)
    {
       h_Tran_Coef[i+6]=0;
    }
  }   
  else
  {
    for (int i=0;i<9;i++)
    {
       h_Tran_Coef[i]=*(R+i);
    }   
    for (int i=0;i<3;i++)
    {
       h_Tran_Coef[i+9]=*(t+i);
    }   
  }
  check(cudaMemcpyToSymbol(d_Tran_Coef, h_Tran_Coef, MAX_SIZE_TRAN * sizeof(double)));
  check(cudaDeviceSynchronize());
}


__global__ void aa(float *d_X,float *d_X_C,int N,int M,int D)
{
       printf("*d_X_C[0] %f \n", *d_X_C);
       printf("*d_X_C[1] %f \n", *(d_X_C+1)); 
       N=10000;
       N=d_Data_Coef[0];
       printf("*N= %d \n", N);
       printf("*M= %d \n", M); 
       printf("*D= %d \n", D);  
       printf("R[1]= %f \n",d_Tran_Coef[0]);
       printf("R[2]= %f \n",d_Tran_Coef[1]);
       printf("t[1]= %f \n",d_Tran_Coef[4]);  
       printf("t[2]= %f \n",d_Tran_Coef[5]);              
       __syncthreads();
       
}

__forceinline__ __device__ unsigned dynamic_smem_size()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}


__global__ void coef2D(float *d_X,float *d_X_C,float *det_X,float *d_Y,float *d_Y_C,float *det_Y,double *B_C)
{
      int n_c=46;
      int ix,iy,N,M,D;
      ix=blockIdx.x*blockDim.x+threadIdx.x;
      iy=blockIdx.y*blockDim.y+threadIdx.y;
      N=d_Data_Coef[0];
      M=d_Data_Coef[1];
      D=d_Data_Coef[2];
 
      double sig;
      sig=d_Sig_Coef[0];
      sig=1/sig;
      float Isig;
      Isig=__double2float_rd(sig);

      __shared__ float X[64],X_C[128],detX[32],Y[64],Y_C[128],detY[32];
      // put the data related to x,y into the shared memory
      if ((threadIdx.y==0) && (ix<N))
      {
         X[threadIdx.x*D]=*(d_X+ix*D);
         //printf("X[%d]= %f \n",ix*D,X[threadIdx.x*D]);
         X[threadIdx.x*D+1]=*(d_X+ix*D+1);
         //printf("X[%d]= %f \n",ix*D+1,X[threadIdx.x*D+1]);
         X_C[threadIdx.x*D*D]=*(d_X_C+ix*D*D)*Isig;
         //printf("X_C[%d]= %f \n",threadIdx.x*D*D,X_C[threadIdx.x*D*D]);
         X_C[threadIdx.x*D*D+1]=*(d_X_C+ix*D*D+1)*Isig;
         //printf("X_C[%d]= %f \n",threadIdx.x*D*D+1,X_C[threadIdx.x*D*D+1]);
         X_C[threadIdx.x*D*D+2]=*(d_X_C+ix*D*D+2)*Isig;
         //printf("X_C[%d]= %f \n",threadIdx.x*D*D+2,X_C[threadIdx.x*D*D+2]);
         X_C[threadIdx.x*D*D+3]=*(d_X_C+ix*D*D+3)*Isig;
         //printf("X_C[%d]= %f \n",threadIdx.x*D*D+3,X_C[threadIdx.x*D*D+3]);
         detX[threadIdx.x]=*(det_X+ix)*Isig;  
         //printf("det_X[%d]= %f \n",threadIdx.x,detX[threadIdx.x]);     
      }

      if ((threadIdx.y==0)  && (blockIdx.y*blockDim.y+threadIdx.x<M))
      {
         Y[threadIdx.x*D]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D,Y[threadIdx.x*D]);         
         Y[threadIdx.x*D+1]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+1);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D+1,Y[threadIdx.x*D+1]); 
         Y_C[threadIdx.x*D*D]=*(d_Y_C+blockIdx.y*blockDim.y*D*D+threadIdx.x*D*D)*Isig; 
         //printf("Y_C[%d]= %f \n",threadIdx.x*D*D,Y_C[threadIdx.x*D*D]);
         Y_C[threadIdx.x*D*D+1]=*(d_Y_C+blockIdx.y*blockDim.y*D*D+threadIdx.x*D*D+1)*Isig;
         //printf("Y_C[%d]= %f \n",threadIdx.x*D*D+1,Y_C[threadIdx.x*D*D+1]);
         Y_C[threadIdx.x*D*D+2]=*(d_Y_C+blockIdx.y*blockDim.y*D*D+threadIdx.x*D*D+2)*Isig;
         //printf("Y_C[%d]= %f \n",threadIdx.x*D*D+2,Y_C[threadIdx.x*D*D+2]);
         Y_C[threadIdx.x*D*D+3]=*(d_Y_C+blockIdx.y*blockDim.y*D*D+threadIdx.x*D*D+3)*Isig;
         //printf("Y_C[%d]= %f \n",threadIdx.x*D*D+3,Y_C[threadIdx.x*D*D+3]);
         detY[threadIdx.x]=*(det_Y+blockIdx.y*blockDim.y+threadIdx.x)*Isig;
         //printf("detY[%d]= %f \n",threadIdx.x,detY[threadIdx.x]);
      }             
      __syncthreads();
      
      __shared__ double mat[32*times2D][46]; 

     // initialize all the elements in mat to zero
     if (threadIdx.x*2<n_c)
     {
        //if time2D==1
        mat[threadIdx.y][threadIdx.x*2]=0;
        //printf("hello %f \n",mat[threadIdx.y][threadIdx.x*2]);
        mat[threadIdx.y][threadIdx.x*2+1]=0;
       // printf("hello %f \n",mat[threadIdx.y][threadIdx.x*2+1]);
     }                                                                                                                    
     __syncthreads();
     
    
     //calculate the coefficient

     if (ix<N && iy<M)
     {
       float n1,n2,m1,m2,a11,a12,a21,a22,b11,b12,b21,b22;
       n1=X[2*threadIdx.x];
       n2=X[2*threadIdx.x+1];
       m1=Y[2*threadIdx.y];
       m2=Y[2*threadIdx.y+1];
       // [a11,a12;a21,a22] is inv(Cy)
       // [b11,b12;b21,b22] is inv(Cx)
       a11=Y_C[4*threadIdx.y];
       a21=Y_C[4*threadIdx.y+1];
       a12=Y_C[4*threadIdx.y+2];
       a22=Y_C[4*threadIdx.y+3];
       b11=X_C[4*threadIdx.x];
       b21=X_C[4*threadIdx.x+1];
       b12=X_C[4*threadIdx.x+2];
       b22=X_C[4*threadIdx.x+3];

       double r11,r21,r12,r22,t1,t2; // the old R,t
       r11=d_Tran_Coef[0];
       r21=d_Tran_Coef[1];
       r12=d_Tran_Coef[2];
       r22=d_Tran_Coef[3];
       t1=d_Tran_Coef[4];
       t2=d_Tran_Coef[5];

       // calculate P
       double P=1;
       double dd1,dd2,c11,c12,c21,c22,mal1,mal2;

       //calculate the P
       dd1=n1 - t1 - m1*r11 - m2*r12;
       dd2=n2 - t2 - m1*r21 - m2*r22;
       // [c11,c12;c21,c22]=R*inv(Cy)*R'
       c11=a11*r11*r11 + a22*r12*r12 + a12*r11*r12 + a21*r11*r12;
       c12=a11*r11*r21 + a12*r11*r22 + a21*r12*r21 + a22*r12*r22;
       c21=a11*r11*r21 + a12*r12*r21 + a21*r11*r22 + a22*r12*r22;
       c22=a11*r21*r21 + a22*r22*r22 + a12*r21*r22 + a21*r21*r22;
       mal1=c11*dd1*dd1 + c22*dd2*dd2 + c12*dd1*dd2 + c21*dd1*dd2;
       
       // calculate (xn-ym)'*inv(Cx)*(xn-ym)
       mal2=b11*dd1*dd1 + b22*dd2*dd2 + b12*dd1*dd2 + b21*dd1*dd2;
       
       // 
       P=detX[threadIdx.x]*detY[threadIdx.y]*(exp(-0.5*mal1)+exp(-0.5*mal2));
       // sum(Q)=1;
       //P=1;

       // update the ym to calculate the P
       dd1=r11*m1+r12*m2+t1;  // the new ym_x
       dd2=r21*m1+r22*m2+t2;  // the new ym_y
       c11= a11*r11*r11 + a22*r12*r12 + a12*r11*r12 + a21*r11*r12; // the new Cy(1,1)
       c12=a11*r11*r21 + a12*r11*r22 + a21*r12*r21 + a22*r12*r22;  // the new Cy(1,2)
       c21=a11*r11*r21 + a12*r12*r21 + a21*r11*r22 + a22*r12*r22;  // the new Cy(2,1)
       c22=a11*r21*r21 + a22*r22*r22 + a12*r21*r22 + a21*r21*r22;  // the new Cy(2,2)
       m1=dd1;
       m2=dd2;
       a11=c11;
       a12=c12;
       a21=c21;
       a22=c22;
       

       __syncthreads();

       //calculate the coefficient;
       double temp(0);

       int i=0;
       temp=a11*m1*m1+ a22*m2*m2+ a12*m1*m2 + a21*m1*m2;
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=1;
       temp=2*a11*m1 + a12*m2 + a21*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=2;
       temp=a12*m1 + a21*m1 + 2*a22*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=3;
       temp=-1* 2*a11*m1*n1- a12*m1*n2 - a12*m2*n1- a21*m1*n2 - a21*m2*n1- 2*a22*m2*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=4;
       temp=2*a11*m1*m1+ 2*a22*m2*m2+ 2*a21*m1*m2+ 2*a12*m1*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=5;
       temp=-1* 2*a22*m2 - a21*m1 - a12*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=6;
       temp=a21*m2 + a12*m2 + 2*a11*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=7;
       temp=2*a22*m2*n1 - a21*m2*n2 + a21*m1*n1 - a12*m2*n2 + a12*m1*n1 - 2*a11*m1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=8;
       temp= a11;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=9;
       temp= a12 + a21;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);


       i=10;
       temp=-1* 2*a11*n1 - a12*n2 - a21*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=11;
       temp=a22;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=12;
       temp=-1* a12*n1 - a21*n1 - 2*a22*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=13;
       temp=a11*n1*n1+ b11*m1*m1+ a22*n2*n2+ b22*m2*m2+ b12*m1*m2  + b21*m1*m2+ a12*n1*n2 + a21*n1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=14;
       temp= a21*m2 + a12*m2 + 2*a11*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=15;
       temp=2*a22*m2+ a21*m1 + a12*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=16;
       temp=-1* 2*a22*m2*n2- a21*m2*n1- a21*m1*n2- a12*m2*n1 - a12*m1*n2- 2*a11*m1*n1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=17;
       temp=-1*a21- a12;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=18;
       temp=-1* 2*a22 + 2*a11;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);


       i=19;
       temp=2*a22*n2 + 2*a21*n1 + 2*a12*n1 - 2*a11*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=20;
       temp= a21+ a12;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=21;
       temp=-1*2*a21*n2 + 2*a22*n1 - 2*a12*n2 - 2*a11*n1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=22;
       temp=-1* a12*n1*n1 + b12*m1 *m1+ a12*n2*n2 - b12*m2*m2- a21*n1*n1 + b21*m1*m1+ a21*n2*n2 - b21*m2*m2 - 2*b11*m1*m2 + 2*b22*m1*m2 + 2*a11*n1*n2 - 2*a22*n1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=23;
       temp=2*b11*m1+ b12*m2+ b21*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=24;
       temp=b12*m1+ b21*m1+ 2*b22*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=25;
       temp=-1* 2*b11*m1*n1- b12*m1*n2- b12*m2*n1- b21*m1*n2- b21*m2*n1- 2*b22*m2*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=26;
       temp=a11*m1*m1+ a22*m2*m2+ a12*m1*m2 + a21*m1*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=27;
       temp=-1* a12*m1 - a21*m1 - 2*a22*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);


       i=28;
       temp=2*a11*m1 + a12*m2 + a21*m2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=29;
       temp=-1* 2*a11*m1*n2+ a12*m1*n1- a12*m2*n2+ a21*m1*n1- a21*m2*n2+ 2*a22*m2*n1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=30;
       temp=a22;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=31;
       temp=-1* a12 - a21;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=32;
       temp=-1* 2*a22*n1 + a21*n2 + a12*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=33;
       temp=a11;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=34;
       temp=a21*n1 + a12*n1 - 2*a11*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=35;
       temp=a11*n2*n2+ b11*m2*m2+ a22*n1*n1+ b22*m1*m1- b12*m1*m2- b21*m1*m2- a12*n1*n2- a21*n1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=36;
       temp=-1* 2*b11*m2+ b12*m1+ b21*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);


       i=37;
       temp=-1* b12*m2- b21*m2+ 2*b22*m1;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=38;
       temp=2*b11*m2*n1- b12*m1*n1+ b12*m2*n2- b21*m1*n1 + b21*m2*n2 - 2*b22*m1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=39;
       temp=b11;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=40;
       temp=b12+ b21;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=41;
       temp=-1* 2*b11*n1- b12*n2- b21*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=42;
       temp=b22;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=43;
       temp=-1* b12*n1- b21*n1- 2*b22*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

       i=44;
       temp=b11*n1*n1 + b22*n2*n2+ b12*n1*n2+ b21*n1*n2;              
       atomicAdd(&mat[threadIdx.x][i],P*temp);

     }















     __syncthreads();     
    
     //check whether the calculation in the last step is correct or not
     //if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.y==0 && threadIdx.x*2<44)
     //{
     //   printf("mat[%d][%d] %f \n",threadIdx.y,threadIdx.x*2,mat[threadIdx.y][threadIdx.x*2]);
     //   printf("mat[%d][%d] %f \n",threadIdx.y,threadIdx.x*2+1,mat[threadIdx.y][threadIdx.x*2+1]);
     //}      
     //__syncthreads();  

     // copy the coefficient matrix in each block to the global memory
     if (D*threadIdx.x<n_c)
     {
       double index1;
       index1=blockIdx.y*gridDim.x*(32*n_c)+blockIdx.x*(32*n_c)+threadIdx.y*n_c+threadIdx.x*D;
       index1=fmod(index1,4096.0*32.0*46.0);
       int index;
       index=__double2int_rn(index1);
       atomicAdd((B_C+index),mat[threadIdx.y][D*threadIdx.x]);
       atomicAdd((B_C+index+1),mat[threadIdx.y][D*threadIdx.x+1]);      
       //printf("B_C[%d]= %f  \n",index % n_c,*(B_C+index)); 
       //printf("B_C[%d]= %f  \n",(index+1)%n_c,*(B_C+index+1));          
     }
     __syncthreads(); 
      
}



__global__ void compi_redu(double * B_C,int n_c, int L,int stride)
{
   int idx=blockIdx.x*blockDim.x+threadIdx.x;
   if (idx<(L+1)/2)
     {
     if (((L&1)==0) || (idx!=(L+1)/2-1))
      {
        for (int i=0;i<n_c;i++)
          atomicAdd((B_C+idx*2*stride+i),*(B_C+idx*2*stride+stride+i));
      }
      } 
   __syncthreads();
   //printf("B_C[0]= %f \n",*B_C);
}


void Redu(double *d_idata,double *coefficient,int size, int n_c)
{
    int t=size/n_c;
    int stride=n_c;
    int blocksize=1024;
    dim3 block(blocksize,1);
    while (t>1) 
    {
       // there are t numbers to deal with
       dim3 grid(((t+1)/2+block.x-1)/block.x,1);
       // invoke the kernel
       compi_redu<<<grid,block>>>(d_idata,n_c,t,stride);
       cudaDeviceSynchronize();
       //update the state
       t=(t+1)/2;
       stride=stride*2;
    }
    int bytes=n_c*sizeof(double);
    cudaMemcpy(coefficient,d_idata,bytes,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //for (int i=0; i<n_c; i++)
    //{
    //   printf(" coefficient[%d] = %f \n",i,*(coefficient+i));
    //}
}




__global__ void coef3D(float *d_X,float *d_X_C,float *det_X,float *d_Y,float *d_Y_C,float *det_Y,double *B_C);

void cpd(
        float *Ptr[6],double *R, double *t, double * N1, double * M1, double * D1,double * sig1, double cMatrix[640]
        )

{ 

  float *d_X, *d_X_C,*d_Y,*d_Y_C, *det_X, *det_Y;
  d_X=Ptr[0];
  d_X_C=Ptr[1];
  d_Y=Ptr[2];
  d_Y_C=Ptr[3];
  det_X=Ptr[4];
  det_Y=Ptr[5];

  int M,N,D;
  N = static_cast<int>(*N1);
  M = static_cast<int>(*M1); 
  D = static_cast<int>(*D1);

  double sig;
  sig=(*sig1);

  setup_coef_constant(N, M, D,sig, R,t);
  
  if (M>3000000 || N>3000000)
  {
     printf("The Number of the Points Exceeds the Maximum Threshold");
     exit(EXIT_FAILURE);
  }
  
  
  int NN,MM;  // the number of coefficient blocks in x, y direction
  unsigned int nBytes;
  // malloc device global memory for coefficient
     // for 2D, we also use the block(32,32), coefficient matrix in shared memory will be [32*45]
     // the maximum number of coefficient matrixes(4*637 or 32*45) is 4096, 3D:80MB, 2D:46MB
  NN=(N+32-1)/(32);  //blocks in X direction
  MM=(M+32-1)/(32);  //blocks in Y direction

  int size,n_c;
  if (D==2)
  { 
     n_c=46;
     if (NN*MM<=4096)
     {
       nBytes=NN*MM*32*n_c*sizeof(double)*times2D;
       //the length of B_C
       size=NN*MM*32*n_c*times2D;
     }
     else 
     {
       nBytes=4096*32*n_c*sizeof(double)*times2D;
       size=4096*32*n_c*times2D;
     }
  }
  else
  {
     n_c=640;
     if (NN*MM<=4096)
     {
       nBytes=NN*MM*4*n_c*sizeof(double)*times3D;
       size=NN*MM*4*n_c*times3D;
     }
     else
     { 
       nBytes=4096*4*n_c*sizeof(double)*times3D;
       size=4096*4*n_c*times3D;
     }     
  }
  double *B_C;  // the coefficient box in device global memory
  check(cudaMalloc((void **)&B_C, nBytes));
  check(cudaMemset(B_C, 0, nBytes));
  check(cudaDeviceSynchronize());

  dim3 block(32,32,1);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
  check(cudaDeviceSynchronize());

  // malloc the memory for the coefficient
  int bytes=n_c*sizeof(double);
  double *coefficient=(double *) malloc(bytes);
  memset(coefficient, 0, bytes);

  if (D==2)
  {
     //coef2D
     coef2D<<<grid,block>>>(d_X,d_X_C,det_X,d_Y,d_Y_C,det_Y,B_C);

     check(cudaDeviceSynchronize());
     //reduction
     Redu(B_C,coefficient,size, n_c);
  }
  else
  {
     //coef3D
     coef3D<<<grid,block>>>(d_X,d_X_C,det_X,d_Y,d_Y_C,det_Y,B_C);

     check(cudaDeviceSynchronize());
     //reduction
     Redu(B_C,coefficient,size, n_c);
  }  
  
  check(cudaDeviceSynchronize());
  
  for (int i=0;i<n_c;i++)
  {
    cMatrix[i]=*(coefficient+i);
  }


  check(cudaDeviceSynchronize());

  //aa<<<1,1>>>(d_X,d_X_C,N,M,D);
  //check(cudaDeviceSynchronize());
  free(coefficient);
  check(cudaFree(B_C));
  return;
}



__global__ void coef3D(float *d_X,float *d_X_C,float *det_X,float *d_Y,float *d_Y_C,float *det_Y,double *B_C)
{
      int n_c=640;
      int ix,iy,N,M,D;
      ix=blockIdx.x*blockDim.x+threadIdx.x;
      iy=blockIdx.y*blockDim.y+threadIdx.y;
      N=d_Data_Coef[0];
      M=d_Data_Coef[1];
      D=d_Data_Coef[2];
 
      double sig;
      sig=d_Sig_Coef[0];
      sig=1/sig;
      float Isig;
      Isig=__double2float_rn(sig);

      __shared__ float X[96],X_C[288],detX[32],Y[96],Y_C[288],detY[32];
      // put the data related to x,y into the shared memory
      if ((threadIdx.y==0) && (ix<N))
      {
         X[threadIdx.x*D]=*(d_X+ix*D);
         //printf("X[%d]= %f \n",ix*D,X[threadIdx.x*D]);
         X[threadIdx.x*D+1]=*(d_X+ix*D+1);
         //printf("X[%d]= %f \n",ix*D+1,X[threadIdx.x*D+1]);
         X[threadIdx.x*D+2]=*(d_X+ix*D+2);
         for (int ii=0;ii<9;ii++)
         {
             X_C[threadIdx.x*D*D+ii]=*(d_X_C+ix*D*D+ii)*Isig;
         }
         detX[threadIdx.x]=*(det_X+ix)*sqrtf(Isig*Isig*Isig);  
         //printf("det_X[%d]= %f \n",threadIdx.x,detX[threadIdx.x]);     
      }
      __syncthreads();
      if ((threadIdx.y==0)  && (blockIdx.y*blockDim.y+threadIdx.x<M))
      {
         Y[threadIdx.x*D]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D,Y[threadIdx.x*D]);         
         Y[threadIdx.x*D+1]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+1);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D+1,Y[threadIdx.x*D+1]); 
         Y[threadIdx.x*D+2]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+2);
         for (int ii=0;ii<9;ii++)
         {
            Y_C[threadIdx.x*D*D+ii]=*(d_Y_C+blockIdx.y*blockDim.y*D*D+threadIdx.x*D*D+ii)*Isig;
         }         

         detY[threadIdx.x]=*(det_Y+blockIdx.y*blockDim.y+threadIdx.x)*sqrtf(Isig*Isig*Isig);
         //printf("detY[%d]= %f \n",threadIdx.x,detY[threadIdx.x]);
      }             
      __syncthreads();
      
      __shared__ double mat[4*times3D][640];

     // initilize mat to zero
 
     if (threadIdx.y<4)
     {
        for (int ii=0;ii<20;ii++)
        {
           mat[threadIdx.y][threadIdx.x*20+ii]=0;
        }
     } 
                                                                                                          
     __syncthreads();
     
    
     //calculate the coefficient

     if (ix<N && iy<M)
     {
       float n1,n2,n3,m1,m2,m3;
       float a11,a12,a13,a21,a22,a23,a31,a32,a33;
       float b11,b12,b13,b21,b22,b23,b31,b32,b33;
       n1=X[3*threadIdx.x];
       n2=X[3*threadIdx.x+1];
       n3=X[3*threadIdx.x+2];       
       m1=Y[3*threadIdx.y];
       m2=Y[3*threadIdx.y+1];
       m3=Y[3*threadIdx.y+2];
       // A is inv(Cy)
       // B is inv(Cx)
       a11=Y_C[9*threadIdx.y];
       a21=Y_C[9*threadIdx.y+1];
       a31=Y_C[9*threadIdx.y+2];
       a12=Y_C[9*threadIdx.y+3];
       a22=Y_C[9*threadIdx.y+4];
       a32=Y_C[9*threadIdx.y+5];
       a13=Y_C[9*threadIdx.y+6]; 
       a23=Y_C[9*threadIdx.y+7];
       a33=Y_C[9*threadIdx.y+8];

       b11=X_C[9*threadIdx.x];
       b21=X_C[9*threadIdx.x+1];
       b31=X_C[9*threadIdx.x+2];
       b12=X_C[9*threadIdx.x+3];
       b22=X_C[9*threadIdx.x+4];
       b32=X_C[9*threadIdx.x+5];
       b13=X_C[9*threadIdx.x+6];
       b23=X_C[9*threadIdx.x+7];
       b33=X_C[9*threadIdx.x+8];

       double r11,r12,r13,r21,r22,r23,r31,r32,r33,t1,t2,t3; // the old R,t
       r11=d_Tran_Coef[0];
       r21=d_Tran_Coef[1];
       r31=d_Tran_Coef[2];
       r12=d_Tran_Coef[3];
       r22=d_Tran_Coef[4];
       r32=d_Tran_Coef[5];
       r13=d_Tran_Coef[6];
       r23=d_Tran_Coef[7];
       r33=d_Tran_Coef[8];

       t1=d_Tran_Coef[9];
       t2=d_Tran_Coef[10];
       t3=d_Tran_Coef[11];

       // calculate P
       double P=1;
       double dd1,dd2,dd3;
       double c11,c12,c13,c21,c22,c23,c31,c32,c33,mal1,mal2;

       //calculate the P
       dd1=n1 - t1 - m1*r11 - m2*r12-m3*r13;
       dd2=n2 - t2 - m1*r21 - m2*r22-m3*r23;
       dd3=n3 - t3 - m1*r31 - m2*r32 - m3*r33;
       // [c11,c12;c21,c22]=R*inv(Cy)*R'
       c11=r11*(a11*r11 + a21*r12 + a31*r13) + r12*(a12*r11 + a22*r12 + a32*r13) + r13*(a13*r11 + a23*r12 + a33*r13);
       c12=r11*(a11*r21 + a21*r22 + a31*r23) + r12*(a12*r21 + a22*r22 + a32*r23) + r13*(a13*r21 + a23*r22 + a33*r23);
       c13=r11*(a11*r31 + a21*r32 + a31*r33) + r12*(a12*r31 + a22*r32 + a32*r33) + r13*(a13*r31 + a23*r32 + a33*r33);
       c21=r21*(a11*r11 + a21*r12 + a31*r13) + r22*(a12*r11 + a22*r12 + a32*r13) + r23*(a13*r11 + a23*r12 + a33*r13);
       c22=r21*(a11*r21 + a21*r22 + a31*r23) + r22*(a12*r21 + a22*r22 + a32*r23) + r23*(a13*r21 + a23*r22 + a33*r23);
       c23=r21*(a11*r31 + a21*r32 + a31*r33) + r22*(a12*r31 + a22*r32 + a32*r33) + r23*(a13*r31 + a23*r32 + a33*r33);
       c31=r31*(a11*r11 + a21*r12 + a31*r13) + r32*(a12*r11 + a22*r12 + a32*r13) + r33*(a13*r11 + a23*r12 + a33*r13);
       c32=r31*(a11*r21 + a21*r22 + a31*r23) + r32*(a12*r21 + a22*r22 + a32*r23) + r33*(a13*r21 + a23*r22 + a33*r23);
       c33=r31*(a11*r31 + a21*r32 + a31*r33) + r32*(a12*r31 + a22*r32 + a32*r33) + r33*(a13*r31 + a23*r32 + a33*r33);

       // calculate (xn-ym)'*inv(R*Cy*R')*(xn-ym); (xn-ym)'*inv(Cx)*(xn-ym)
       mal1=dd1*(c11*dd1 + c21*dd2 + c31*dd3) + dd2*(c12*dd1 + c22*dd2 + c32*dd3) + dd3*(c13*dd1 + c23*dd2 + c33*dd3);
       mal2=dd1*(b11*dd1 + b21*dd2 + b31*dd3) + dd2*(b12*dd1 + b22*dd2 + b32*dd3) + dd3*(b13*dd1 + b23*dd2 + b33*dd3);
       
       // 
       P=detX[threadIdx.x]*detY[threadIdx.y]*(exp(-0.5*mal1)+exp(-0.5*mal2));

       // update the ym to calculate the P
       dd1=r11*m1+r12*m2+r13*m3+t1;  // the new ym_x
       dd2=r21*m1+r22*m2+r23*m3+t2;  // the new ym_y
       dd3=r31*m1+r32*m2+r33*m3+t3;
       

       m1=dd1;
       m2=dd2;
       m3=dd3;
       a11=c11;
       a12=c12;
       a13=c13;
       a21=c21;
       a22=c22;
       a23=c23;
       a31=c31;
       a32=c32;
       a33=c33;
       

       __syncthreads();

       //calculate the coefficient;
       double temp(0);
       int i;

i=0;
temp= b11*n1*n1 + b22*n2*n2 + b33*n3*n3 + b12*n1*n2 + b13*n1*n3 + b21*n1*n2 + b23*n2*n3 + b31*n1*n3 + b32*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);
     __syncthreads(); 

i=1;
temp= a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);
 

i=2;
temp= a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);


i=3;
temp= a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);
 

i=4;
temp= a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=5;
temp= a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=6;
temp= a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=7;
temp= a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=8;
temp= a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=9;
temp= a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=10;
temp= a11*n1*n1 + b11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=11;
temp= b11*m2*m2 + a22*n1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=12;
temp= b11*m3*m3 + a33*n1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=13;
temp= a11*n2*n2 + b22*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=14;
temp= a11*n3*n3 + b33*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=15;
temp= a22*n2*n2 + b22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=16;
temp= b22*m3*m3 + a33*n2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=17;
temp= a22*n3*n3 + b33*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=18;
temp= a33*n3*n3 + b33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=19;
temp= (-1)*2*b11*m1*n1 + (-1)*b12*m1*n2 + (-1)*b13*m1*n3 + (-1)*b21*m1*n2 + (-1)*b31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=20;
temp= (-1)*2*b11*m2*n1 + (-1)*b12*m2*n2 + (-1)*b13*m2*n3 + (-1)*b21*m2*n2 + (-1)*b31*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=21;
temp= (-1)*2*b11*m3*n1 + (-1)*b12*m3*n2 + (-1)*b13*m3*n3 + (-1)*b21*m3*n2 + (-1)*b31*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=22;
temp= (-1)*b12*m1*n1 + (-1)*b21*m1*n1 + (-1)*2*b22*m1*n2 + (-1)*b23*m1*n3 + (-1)*b32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=23;
temp= (-1)*b12*m2*n1 + (-1)*b21*m2*n1 + (-1)*2*b22*m2*n2 + (-1)*b23*m2*n3 + (-1)*b32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=24;
temp= (-1)*b12*m3*n1 + (-1)*b21*m3*n1 + (-1)*2*b22*m3*n2 + (-1)*b23*m3*n3 + (-1)*b32*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=25;
temp= (-1)*b13*m1*n1 + (-1)*b23*m1*n2 + (-1)*b31*m1*n1 + (-1)*b32*m1*n2 + (-1)*2*b33*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=26;
temp= (-1)*b13*m2*n1 + (-1)*b23*m2*n2 + (-1)*b31*m2*n1 + (-1)*b32*m2*n2 + (-1)*2*b33*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=27;
temp= (-1)*b13*m3*n1 + (-1)*b23*m3*n2 + (-1)*b31*m3*n1 + (-1)*b32*m3*n2 + (-1)*2*b33*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=28;
temp= a11*m2*m2 + a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=29;
temp= a11*m3*m3 + a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=30;
temp= 2*a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=31;
temp= a22*m3*m3 + a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=32;
temp= 2*a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=33;
temp= a11*m2*m2 + a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=34;
temp= a11*m3*m3 + a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=35;
temp= 2*a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=36;
temp= 2*a11*m1*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=37;
temp= 2*a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=38;
temp= a22*m3*m3 + a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=39;
temp= 2*a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=40;
temp= a11*m2*m2 + a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=41;
temp= a11*m3*m3 + a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=42;
temp= 2*a22*m2*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=43;
temp= 2*a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=44;
temp= a22*m3*m3 + a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=45;
temp= 2*a33*m3*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=46;
temp= (-1)*2*a11*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=47;
temp= (-1)*2*a11*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=48;
temp= (-1)*2*a22*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=49;
temp= (-1)*2*a11*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=50;
temp= (-1)*2*a22*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=51;
temp= (-1)*2*a33*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=52;
temp= (-1)*2*a22*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=53;
temp= (-1)*2*a33*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=54;
temp= (-1)*2*a33*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=55;
temp= a12*n1*n1 + a21*n1*n1 + 2*b11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=56;
temp= a13*n1*n1 + a31*n1*n1 + 2*b11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=57;
temp= b12*m1*m1 + b21*m1*m1 + 2*a11*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=58;
temp= b12*m2*m2 + b21*m2*m2 + 2*a22*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=59;
temp= a23*n1*n1 + a32*n1*n1 + 2*b11*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=60;
temp= b12*m3*m3 + b21*m3*m3 + 2*a33*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=61;
temp= b13*m1*m1 + b31*m1*m1 + 2*a11*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=62;
temp= a12*n2*n2 + a21*n2*n2 + 2*b22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=63;
temp= a13*n2*n2 + a31*n2*n2 + 2*b22*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=64;
temp= b13*m2*m2 + b31*m2*m2 + 2*a22*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=65;
temp= b13*m3*m3 + b31*m3*m3 + 2*a33*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=66;
temp= a23*n2*n2 + a32*n2*n2 + 2*b22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=67;
temp= b23*m1*m1 + b32*m1*m1 + 2*a11*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=68;
temp= a12*n3*n3 + a21*n3*n3 + 2*b33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=69;
temp= b23*m2*m2 + b32*m2*m2 + 2*a22*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=70;
temp= a13*n3*n3 + a31*n3*n3 + 2*b33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=71;
temp= b23*m3*m3 + b32*m3*m3 + 2*a33*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=72;
temp= a23*n3*n3 + a32*n3*n3 + 2*b33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=73;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=74;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=75;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=76;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=77;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=78;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=79;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=80;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=81;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=82;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=83;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=84;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=85;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=86;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=87;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=88;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=89;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=90;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=91;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=92;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=93;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=94;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=95;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=96;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=97;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=98;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=99;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=100;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=101;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=102;
temp= a12*m1*m2 + a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=103;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=104;
temp= a13*m1*m3 + a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=105;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=106;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=107;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=108;
temp= a23*m2*m3 + a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=109;
temp= a12*m3*m3 + a21*m3*m3 + 2*a13*m2*m3 + 2*a23*m1*m3 + 2*a31*m2*m3 + 2*a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=110;
temp= a13*m2*m2 + a31*m2*m2 + 2*a12*m2*m3 + 2*a21*m2*m3 + 2*a22*m1*m3 + 2*a23*m1*m2 + 2*a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=111;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=112;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=113;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=114;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + 2*a12*m1*m3 + 2*a13*m1*m2 + 2*a21*m1*m3 + 2*a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=115;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=116;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=117;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=118;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=119;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=120;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=121;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=122;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=123;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=124;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=125;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=126;
temp= a12*m3*m3 + a21*m3*m3 + 2*a13*m2*m3 + 2*a23*m1*m3 + 2*a31*m2*m3 + 2*a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=127;
temp= a13*m2*m2 + a31*m2*m2 + 2*a12*m2*m3 + 2*a21*m2*m3 + 2*a22*m1*m3 + 2*a23*m1*m2 + 2*a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=128;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=129;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=130;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=131;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=132;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=133;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=134;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=135;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=136;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=137;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=138;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + 2*a12*m1*m3 + 2*a13*m1*m2 + 2*a21*m1*m3 + 2*a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=139;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=140;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=141;
temp= a12*m1*m1 + a21*m1*m1 + 2*a11*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=142;
temp= a12*m2*m2 + a21*m2*m2 + 2*a22*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=143;
temp= a13*m1*m1 + a31*m1*m1 + 2*a11*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=144;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=145;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=146;
temp= a13*m3*m3 + a31*m3*m3 + 2*a33*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=147;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=148;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=149;
temp= a12*m3*m3 + a21*m3*m3 + 2*a13*m2*m3 + 2*a23*m1*m3 + 2*a31*m2*m3 + 2*a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=150;
temp= a13*m2*m2 + a31*m2*m2 + 2*a12*m2*m3 + 2*a21*m2*m3 + 2*a22*m1*m3 + 2*a23*m1*m2 + 2*a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=151;
temp= a23*m2*m2 + a32*m2*m2 + 2*a22*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=152;
temp= a23*m3*m3 + a32*m3*m3 + 2*a33*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=153;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + 2*a12*m1*m3 + 2*a13*m1*m2 + 2*a21*m1*m3 + 2*a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=154;
temp= b12*m1*m2 + b21*m1*m2 + a12*n1*n2 + a21*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=155;
temp= b12*m1*m2 + b21*m1*m2 + a12*n1*n2 + a21*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=156;
temp= b12*m1*m3 + b21*m1*m3 + a13*n1*n2 + a31*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=157;
temp= b12*m1*m3 + b21*m1*m3 + a13*n1*n2 + a31*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=158;
temp= b12*m2*m3 + b21*m2*m3 + a23*n1*n2 + a32*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=159;
temp= b12*m2*m3 + b21*m2*m3 + a23*n1*n2 + a32*n1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=160;
temp= b13*m1*m2 + b31*m1*m2 + a12*n1*n3 + a21*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=161;
temp= b13*m1*m2 + b31*m1*m2 + a12*n1*n3 + a21*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=162;
temp= b13*m1*m3 + b31*m1*m3 + a13*n1*n3 + a31*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=163;
temp= b13*m1*m3 + b31*m1*m3 + a13*n1*n3 + a31*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=164;
temp= b13*m2*m3 + b31*m2*m3 + a23*n1*n3 + a32*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=165;
temp= b13*m2*m3 + b31*m2*m3 + a23*n1*n3 + a32*n1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=166;
temp= b23*m1*m2 + b32*m1*m2 + a12*n2*n3 + a21*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=167;
temp= b23*m1*m2 + b32*m1*m2 + a12*n2*n3 + a21*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=168;
temp= b23*m1*m3 + b32*m1*m3 + a13*n2*n3 + a31*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=169;
temp= b23*m1*m3 + b32*m1*m3 + a13*n2*n3 + a31*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=170;
temp= b23*m2*m3 + b32*m2*m3 + a23*n2*n3 + a32*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=171;
temp= b23*m2*m3 + b32*m2*m3 + a23*n2*n3 + a32*n2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=172;
temp= (-1)*2*a11*m2*n1 + (-1)*2*a12*m1*n1 + (-1)*2*a21*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=173;
temp= (-1)*2*a12*m2*n1 + (-1)*2*a21*m2*n1 + (-1)*2*a22*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=174;
temp= (-1)*2*a11*m3*n1 + (-1)*2*a13*m1*n1 + (-1)*2*a31*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=175;
temp= (-1)*2*a13*m3*n1 + (-1)*2*a31*m3*n1 + (-1)*2*a33*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=176;
temp= (-1)*2*a11*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=177;
temp= (-1)*2*a11*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=178;
temp= (-1)*a12*m1*n1 + (-1)*a21*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=179;
temp= (-1)*a12*m1*n2 + (-1)*a21*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=180;
temp= (-1)*a12*m2*n1 + (-1)*a21*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=181;
temp= (-1)*a12*m2*n2 + (-1)*a21*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=182;
temp= (-1)*a13*m1*n1 + (-1)*a31*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=183;
temp= (-1)*a13*m1*n2 + (-1)*a31*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=184;
temp= (-1)*a13*m3*n1 + (-1)*a31*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=185;
temp= (-1)*2*a22*m3*n1 + (-1)*2*a23*m2*n1 + (-1)*2*a32*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=186;
temp= (-1)*a13*m3*n2 + (-1)*a31*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=187;
temp= (-1)*2*a23*m3*n1 + (-1)*2*a32*m3*n1 + (-1)*2*a33*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=188;
temp= (-1)*2*a11*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=189;
temp= (-1)*2*a11*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=190;
temp= (-1)*a12*m1*n1 + (-1)*a21*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=191;
temp= (-1)*2*a11*m2*n2 + (-1)*2*a12*m1*n2 + (-1)*2*a21*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=192;
temp= (-1)*a12*m2*n1 + (-1)*a21*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=193;
temp= (-1)*a12*m1*n3 + (-1)*a21*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=194;
temp= (-1)*2*a12*m2*n2 + (-1)*2*a21*m2*n2 + (-1)*2*a22*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=195;
temp= (-1)*a13*m1*n1 + (-1)*a31*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=196;
temp= (-1)*2*a22*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=197;
temp= (-1)*2*a11*m3*n2 + (-1)*2*a13*m1*n2 + (-1)*2*a31*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=198;
temp= (-1)*a12*m2*n3 + (-1)*a21*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=199;
temp= (-1)*2*a22*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=200;
temp= (-1)*a13*m1*n3 + (-1)*a31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=201;
temp= (-1)*a13*m3*n1 + (-1)*a31*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=202;
temp= (-1)*a23*m2*n1 + (-1)*a32*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=203;
temp= (-1)*2*a13*m3*n2 + (-1)*2*a31*m3*n2 + (-1)*2*a33*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=204;
temp= (-1)*a23*m2*n2 + (-1)*a32*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=205;
temp= (-1)*a23*m3*n1 + (-1)*a32*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=206;
temp= (-1)*a13*m3*n3 + (-1)*a31*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=207;
temp= (-1)*a23*m3*n2 + (-1)*a32*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=208;
temp= (-1)*2*a11*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=209;
temp= (-1)*2*a11*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=210;
temp= (-1)*a12*m1*n2 + (-1)*a21*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=211;
temp= (-1)*a12*m1*n3 + (-1)*a21*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=212;
temp= (-1)*a12*m2*n2 + (-1)*a21*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=213;
temp= (-1)*2*a22*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=214;
temp= (-1)*a12*m2*n3 + (-1)*a21*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=215;
temp= (-1)*a13*m1*n2 + (-1)*a31*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=216;
temp= (-1)*a13*m1*n3 + (-1)*a31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=217;
temp= (-1)*2*a22*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=218;
temp= (-1)*a23*m2*n1 + (-1)*a32*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=219;
temp= (-1)*a13*m3*n2 + (-1)*a31*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=220;
temp= (-1)*2*a22*m3*n2 + (-1)*2*a23*m2*n2 + (-1)*2*a32*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=221;
temp= (-1)*a23*m3*n1 + (-1)*a32*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=222;
temp= (-1)*a13*m3*n3 + (-1)*a31*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=223;
temp= (-1)*a23*m2*n3 + (-1)*a32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=224;
temp= (-1)*2*a23*m3*n2 + (-1)*2*a32*m3*n2 + (-1)*2*a33*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=225;
temp= (-1)*2*a33*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=226;
temp= (-1)*a23*m3*n3 + (-1)*a32*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=227;
temp= (-1)*2*a33*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=228;
temp= (-1)*2*a11*m2*n3 + (-1)*2*a12*m1*n3 + (-1)*2*a21*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=229;
temp= (-1)*2*a12*m2*n3 + (-1)*2*a21*m2*n3 + (-1)*2*a22*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=230;
temp= (-1)*2*a22*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=231;
temp= (-1)*2*a11*m3*n3 + (-1)*2*a13*m1*n3 + (-1)*2*a31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=232;
temp= (-1)*2*a22*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=233;
temp= (-1)*a23*m2*n2 + (-1)*a32*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=234;
temp= (-1)*2*a13*m3*n3 + (-1)*2*a31*m3*n3 + (-1)*2*a33*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=235;
temp= (-1)*a23*m2*n3 + (-1)*a32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=236;
temp= (-1)*a23*m3*n2 + (-1)*a32*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=237;
temp= (-1)*2*a33*m3*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=238;
temp= (-1)*a23*m3*n3 + (-1)*a32*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=239;
temp= (-1)*2*a33*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=240;
temp= (-1)*2*a22*m3*n3 + (-1)*2*a23*m2*n3 + (-1)*2*a32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=241;
temp= (-1)*2*a23*m3*n3 + (-1)*2*a32*m3*n3 + (-1)*2*a33*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=242;
temp= (-1)*2*a33*m3*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=243;
temp= (-1)*2*a33*m3*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=244;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=245;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=246;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=247;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=248;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=249;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=250;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=251;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=252;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=253;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=254;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=255;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=256;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=257;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=258;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=259;
temp= a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=260;
temp= a12*m2*m3 + a21*m2*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=261;
temp= a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=262;
temp= 2*a11*m2*m2 + 2*a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=263;
temp= 2*a11*m3*m3 + 2*a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=264;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=265;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=266;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=267;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=268;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=269;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=270;
temp= 2*a22*m3*m3 + 2*a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=271;
temp= 2*a11*m2*m2 + 2*a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=272;
temp= 2*a11*m3*m3 + 2*a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=273;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=274;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=275;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=276;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=277;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=278;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=279;
temp= 2*a22*m3*m3 + 2*a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=280;
temp= 2*a11*m2*m2 + 2*a22*m1*m1 + 2*a12*m1*m2 + 2*a21*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=281;
temp= 2*a11*m3*m3 + 2*a33*m1*m1 + 2*a13*m1*m3 + 2*a31*m1*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=282;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=283;
temp= a13*m2*m2 + a31*m2*m2 + a12*m2*m3 + a21*m2*m3 + 2*a22*m1*m3 + a23*m1*m2 + a32*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=284;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=285;
temp= a12*m3*m3 + a21*m3*m3 + a13*m2*m3 + a23*m1*m3 + a31*m2*m3 + a32*m1*m3 + 2*a33*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=286;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=287;
temp= a23*m1*m1 + a32*m1*m1 + 2*a11*m2*m3 + a12*m1*m3 + a13*m1*m2 + a21*m1*m3 + a31*m1*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=288;
temp= 2*a22*m3*m3 + 2*a33*m2*m2 + 2*a23*m2*m3 + 2*a32*m2*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=289;
temp= (-1)*2*a12*m3*n1 + (-1)*2*a13*m2*n1 + (-1)*2*a21*m3*n1 + (-1)*2*a23*m1*n1 + (-1)*2*a31*m2*n1 + (-1)*2*a32*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=290;
temp= (-1)*2*a11*m2*n2 + (-1)*a12*m1*n2 + (-1)*a21*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=291;
temp= (-1)*2*a11*m3*n2 + (-1)*a13*m1*n2 + (-1)*a31*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=292;
temp= (-1)*a12*m2*n2 + (-1)*a21*m2*n2 + (-1)*2*a22*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=293;
temp= (-1)*a12*m3*n2 + (-1)*a21*m3*n2 + (-1)*a23*m1*n2 + (-1)*a32*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=294;
temp= (-1)*a12*m3*n2 + (-1)*a13*m2*n2 + (-1)*a21*m3*n2 + (-1)*a31*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=295;
temp= (-1)*a13*m2*n2 + (-1)*a23*m1*n2 + (-1)*a31*m2*n2 + (-1)*a32*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=296;
temp= (-1)*a13*m3*n2 + (-1)*a31*m3*n2 + (-1)*2*a33*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=297;
temp= (-1)*2*a11*m2*n1 + (-1)*a12*m1*n1 + (-1)*a21*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=298;
temp= (-1)*2*a11*m2*n3 + (-1)*a12*m1*n3 + (-1)*a21*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=299;
temp= (-1)*2*a11*m3*n1 + (-1)*a13*m1*n1 + (-1)*a31*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=300;
temp= (-1)*a12*m2*n1 + (-1)*a21*m2*n1 + (-1)*2*a22*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=301;
temp= (-1)*2*a11*m3*n3 + (-1)*a13*m1*n3 + (-1)*a31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=302;
temp= (-1)*a12*m2*n3 + (-1)*a21*m2*n3 + (-1)*2*a22*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=303;
temp= (-1)*a12*m3*n1 + (-1)*a13*m2*n1 + (-1)*a21*m3*n1 + (-1)*a31*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=304;
temp= (-1)*a12*m3*n1 + (-1)*a21*m3*n1 + (-1)*a23*m1*n1 + (-1)*a32*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=305;
temp= (-1)*a13*m2*n1 + (-1)*a23*m1*n1 + (-1)*a31*m2*n1 + (-1)*a32*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=306;
temp= (-1)*a12*m3*n3 + (-1)*a21*m3*n3 + (-1)*a23*m1*n3 + (-1)*a32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=307;
temp= (-1)*a12*m3*n3 + (-1)*a13*m2*n3 + (-1)*a21*m3*n3 + (-1)*a31*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=308;
temp= (-1)*a13*m2*n3 + (-1)*a23*m1*n3 + (-1)*a31*m2*n3 + (-1)*a32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=309;
temp= (-1)*a13*m3*n1 + (-1)*a31*m3*n1 + (-1)*2*a33*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=310;
temp= (-1)*2*a22*m3*n2 + (-1)*a23*m2*n2 + (-1)*a32*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=311;
temp= (-1)*a13*m3*n3 + (-1)*a31*m3*n3 + (-1)*2*a33*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=312;
temp= (-1)*a23*m3*n2 + (-1)*a32*m3*n2 + (-1)*2*a33*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=313;
temp= (-1)*2*a12*m3*n2 + (-1)*2*a13*m2*n2 + (-1)*2*a21*m3*n2 + (-1)*2*a23*m1*n2 + (-1)*2*a31*m2*n2 + (-1)*2*a32*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=314;
temp= (-1)*2*a22*m3*n1 + (-1)*a23*m2*n1 + (-1)*a32*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=315;
temp= (-1)*2*a22*m3*n3 + (-1)*a23*m2*n3 + (-1)*a32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=316;
temp= (-1)*a23*m3*n1 + (-1)*a32*m3*n1 + (-1)*2*a33*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=317;
temp= (-1)*a23*m3*n3 + (-1)*a32*m3*n3 + (-1)*2*a33*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=318;
temp= (-1)*2*a11*m2*n1 + (-1)*a12*m1*n1 + (-1)*a21*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=319;
temp= (-1)*2*a11*m2*n3 + (-1)*a12*m1*n3 + (-1)*a21*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=320;
temp= (-1)*2*a11*m3*n1 + (-1)*a13*m1*n1 + (-1)*a31*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=321;
temp= (-1)*a12*m2*n1 + (-1)*a21*m2*n1 + (-1)*2*a22*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=322;
temp= (-1)*2*a11*m3*n3 + (-1)*a13*m1*n3 + (-1)*a31*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=323;
temp= (-1)*a12*m2*n3 + (-1)*a21*m2*n3 + (-1)*2*a22*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=324;
temp= (-1)*a12*m3*n1 + (-1)*a13*m2*n1 + (-1)*a21*m3*n1 + (-1)*a31*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=325;
temp= (-1)*a12*m3*n1 + (-1)*a21*m3*n1 + (-1)*a23*m1*n1 + (-1)*a32*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=326;
temp= (-1)*a13*m2*n1 + (-1)*a23*m1*n1 + (-1)*a31*m2*n1 + (-1)*a32*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=327;
temp= (-1)*a12*m3*n3 + (-1)*a21*m3*n3 + (-1)*a23*m1*n3 + (-1)*a32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=328;
temp= (-1)*a12*m3*n3 + (-1)*a13*m2*n3 + (-1)*a21*m3*n3 + (-1)*a31*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=329;
temp= (-1)*a13*m2*n3 + (-1)*a23*m1*n3 + (-1)*a31*m2*n3 + (-1)*a32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=330;
temp= (-1)*a13*m3*n1 + (-1)*a31*m3*n1 + (-1)*2*a33*m1*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=331;
temp= (-1)*a13*m3*n3 + (-1)*a31*m3*n3 + (-1)*2*a33*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=332;
temp= (-1)*2*a11*m2*n2 + (-1)*a12*m1*n2 + (-1)*a21*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=333;
temp= (-1)*2*a11*m3*n2 + (-1)*a13*m1*n2 + (-1)*a31*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=334;
temp= (-1)*a12*m2*n2 + (-1)*a21*m2*n2 + (-1)*2*a22*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=335;
temp= (-1)*a12*m3*n2 + (-1)*a13*m2*n2 + (-1)*a21*m3*n2 + (-1)*a31*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=336;
temp= (-1)*a12*m3*n2 + (-1)*a21*m3*n2 + (-1)*a23*m1*n2 + (-1)*a32*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=337;
temp= (-1)*a13*m2*n2 + (-1)*a23*m1*n2 + (-1)*a31*m2*n2 + (-1)*a32*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=338;
temp= (-1)*2*a22*m3*n1 + (-1)*a23*m2*n1 + (-1)*a32*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=339;
temp= (-1)*a13*m3*n2 + (-1)*a31*m3*n2 + (-1)*2*a33*m1*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=340;
temp= (-1)*2*a22*m3*n3 + (-1)*a23*m2*n3 + (-1)*a32*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=341;
temp= (-1)*a23*m3*n1 + (-1)*a32*m3*n1 + (-1)*2*a33*m2*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=342;
temp= (-1)*a23*m3*n3 + (-1)*a32*m3*n3 + (-1)*2*a33*m2*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=343;
temp= (-1)*2*a12*m3*n3 + (-1)*2*a13*m2*n3 + (-1)*2*a21*m3*n3 + (-1)*2*a23*m1*n3 + (-1)*2*a31*m2*n3 + (-1)*2*a32*m1*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=344;
temp= (-1)*2*a22*m3*n2 + (-1)*a23*m2*n2 + (-1)*a32*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=345;
temp= (-1)*a23*m3*n2 + (-1)*a32*m3*n2 + (-1)*2*a33*m2*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=346;
temp= b11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=347;
temp= b22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=348;
temp= b33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=349;
temp= (-1)*2*b11*n1 + (-1)*b12*n2 + (-1)*b13*n3 + (-1)*b21*n2 + (-1)*b31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=350;
temp= (-1)*b12*n1 + (-1)*b21*n1 + (-1)*2*b22*n2 + (-1)*b23*n3 + (-1)*b32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=351;
temp= (-1)*b13*n1 + (-1)*b23*n2 + (-1)*b31*n1 + (-1)*b32*n2 + (-1)*2*b33*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=352;
temp= b12 + b21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=353;
temp= b13 + b31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=354;
temp= b23 + b32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=355;
temp= a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=356;
temp= a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=357;
temp= a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=358;
temp= a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=359;
temp= a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=360;
temp= a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=361;
temp= a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=362;
temp= a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=363;
temp= a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=364;
temp= 2*b11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=365;
temp= 2*b11*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=366;
temp= b12*m1 + b21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=367;
temp= 2*b11*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=368;
temp= b12*m2 + b21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=369;
temp= b13*m1 + b31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=370;
temp= b12*m3 + b21*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=371;
temp= b13*m2 + b31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=372;
temp= b13*m3 + b31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=373;
temp= b12*m1 + b21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=374;
temp= b12*m2 + b21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=375;
temp= b12*m3 + b21*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=376;
temp= b13*m1 + b31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=377;
temp= 2*b22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=378;
temp= b13*m2 + b31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=379;
temp= 2*b22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=380;
temp= b23*m1 + b32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=381;
temp= b13*m3 + b31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=382;
temp= 2*b22*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=383;
temp= b23*m2 + b32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=384;
temp= b23*m3 + b32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=385;
temp= b23*m1 + b32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=386;
temp= b23*m2 + b32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=387;
temp= b23*m3 + b32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=388;
temp= 2*b33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=389;
temp= 2*b33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=390;
temp= 2*b33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=391;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=392;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=393;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=394;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=395;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=396;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=397;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=398;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=399;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=400;
temp= (-1)*2*a11*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=401;
temp= (-1)*2*a11*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=402;
temp= (-1)*2*a22*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=403;
temp= (-1)*2*a11*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=404;
temp= (-1)*2*a22*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=405;
temp= (-1)*2*a33*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=406;
temp= (-1)*2*a22*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=407;
temp= (-1)*2*a33*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=408;
temp= (-1)*2*a33*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=409;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=410;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=411;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=412;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=413;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=414;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=415;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=416;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=417;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=418;
temp= (-1)*2*a12*n1 + (-1)*2*a21*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=419;
temp= (-1)*2*a13*n1 + (-1)*2*a31*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=420;
temp= (-1)*2*a11*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=421;
temp= (-1)*2*a11*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=422;
temp= (-1)*a12*n1 + (-1)*a21*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=423;
temp= (-1)*a12*n1 + (-1)*a21*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=424;
temp= (-1)*a12*n2 + (-1)*a21*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=425;
temp= (-1)*a12*n2 + (-1)*a21*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=426;
temp= (-1)*a13*n1 + (-1)*a31*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=427;
temp= (-1)*a13*n1 + (-1)*a31*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=428;
temp= (-1)*a13*n2 + (-1)*a31*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=429;
temp= (-1)*a13*n2 + (-1)*a31*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=430;
temp= (-1)*2*a23*n1 + (-1)*2*a32*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=431;
temp= (-1)*2*a11*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=432;
temp= (-1)*2*a11*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=433;
temp= (-1)*a12*n1 + (-1)*a21*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=434;
temp= (-1)*a12*n1 + (-1)*a21*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=435;
temp= (-1)*2*a12*n2 + (-1)*2*a21*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=436;
temp= (-1)*a12*n3 + (-1)*a21*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=437;
temp= (-1)*a12*n3 + (-1)*a21*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=438;
temp= (-1)*2*a22*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=439;
temp= (-1)*2*a22*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=440;
temp= (-1)*a13*n1 + (-1)*a31*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=441;
temp= (-1)*a13*n1 + (-1)*a31*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=442;
temp= (-1)*2*a13*n2 + (-1)*2*a31*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=443;
temp= (-1)*a13*n3 + (-1)*a31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=444;
temp= (-1)*a13*n3 + (-1)*a31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=445;
temp= (-1)*a23*n1 + (-1)*a32*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=446;
temp= (-1)*a23*n1 + (-1)*a32*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=447;
temp= (-1)*a23*n2 + (-1)*a32*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=448;
temp= (-1)*a23*n2 + (-1)*a32*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=449;
temp= (-1)*2*a11*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=450;
temp= (-1)*2*a11*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=451;
temp= (-1)*a12*n2 + (-1)*a21*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=452;
temp= (-1)*a12*n2 + (-1)*a21*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=453;
temp= (-1)*a12*n3 + (-1)*a21*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=454;
temp= (-1)*a12*n3 + (-1)*a21*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=455;
temp= (-1)*2*a22*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=456;
temp= (-1)*2*a22*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=457;
temp= (-1)*a13*n2 + (-1)*a31*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=458;
temp= (-1)*a13*n2 + (-1)*a31*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=459;
temp= (-1)*a13*n3 + (-1)*a31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=460;
temp= (-1)*a13*n3 + (-1)*a31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=461;
temp= (-1)*a23*n1 + (-1)*a32*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=462;
temp= (-1)*a23*n1 + (-1)*a32*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=463;
temp= (-1)*2*a23*n2 + (-1)*2*a32*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=464;
temp= (-1)*a23*n3 + (-1)*a32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=465;
temp= (-1)*a23*n3 + (-1)*a32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=466;
temp= (-1)*2*a33*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=467;
temp= (-1)*2*a33*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=468;
temp= (-1)*2*a12*n3 + (-1)*2*a21*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=469;
temp= (-1)*2*a22*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=470;
temp= (-1)*2*a22*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=471;
temp= (-1)*2*a13*n3 + (-1)*2*a31*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=472;
temp= (-1)*a23*n2 + (-1)*a32*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=473;
temp= (-1)*a23*n2 + (-1)*a32*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=474;
temp= (-1)*a23*n3 + (-1)*a32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=475;
temp= (-1)*a23*n3 + (-1)*a32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=476;
temp= (-1)*2*a33*n1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=477;
temp= (-1)*2*a33*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=478;
temp= (-1)*2*a23*n3 + (-1)*2*a32*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=479;
temp= (-1)*2*a33*n2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=480;
temp= (-1)*2*a33*n3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=481;
temp= 2*a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=482;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=483;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=484;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=485;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=486;
temp= 2*a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=487;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=488;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=489;
temp= 2*a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=490;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=491;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=492;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=493;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=494;
temp= 2*a11;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=495;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=496;
temp= a12 + a21;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=497;
temp= 2*a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=498;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=499;
temp= a13 + a31;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=500;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=501;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=502;
temp= 2*a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=503;
temp= 2*a22;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=504;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=505;
temp= a23 + a32;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=506;
temp= 2*a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=507;
temp= 2*a33;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=508;
temp= 2*a11*m2 + 2*a12*m1 + 2*a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=509;
temp= 2*a12*m2 + 2*a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=510;
temp= 2*a11*m3 + 2*a13*m1 + 2*a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=511;
temp= 2*a13*m3 + 2*a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=512;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=513;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=514;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=515;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=516;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=517;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=518;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=519;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=520;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=521;
temp= 2*a22*m3 + 2*a23*m2 + 2*a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=522;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=523;
temp= 2*a23*m3 + 2*a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=524;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=525;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=526;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=527;
temp= 2*a11*m2 + 2*a12*m1 + 2*a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=528;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=529;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=530;
temp= 2*a12*m2 + 2*a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=531;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=532;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=533;
temp= 2*a11*m3 + 2*a13*m1 + 2*a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=534;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=535;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=536;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=537;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=538;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=539;
temp= 2*a13*m3 + 2*a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=540;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=541;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=542;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=543;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=544;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=545;
temp= 2*a11*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=546;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=547;
temp= a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=548;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=549;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=550;
temp= a12*m2 + a21*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=551;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=552;
temp= a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=553;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=554;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=555;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=556;
temp= 2*a22*m3 + 2*a23*m2 + 2*a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=557;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=558;
temp= a13*m3 + a31*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=559;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=560;
temp= 2*a23*m3 + 2*a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=561;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=562;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=563;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=564;
temp= 2*a11*m2 + 2*a12*m1 + 2*a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=565;
temp= 2*a12*m2 + 2*a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=566;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=567;
temp= 2*a11*m3 + 2*a13*m1 + 2*a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=568;
temp= 2*a22*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=569;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=570;
temp= 2*a13*m3 + 2*a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=571;
temp= a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=572;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=573;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=574;
temp= a23*m3 + a32*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=575;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=576;
temp= 2*a22*m3 + 2*a23*m2 + 2*a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=577;
temp= 2*a23*m3 + 2*a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=578;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=579;
temp= 2*a33*m3;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=580;
temp= 2*a12*m3 + 2*a13*m2 + 2*a21*m3 + 2*a23*m1 + 2*a31*m2 + 2*a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=581;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=582;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=583;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=584;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=585;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=586;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=587;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=588;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=589;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=590;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=591;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=592;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=593;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=594;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=595;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=596;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=597;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=598;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=599;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=600;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=601;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=602;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=603;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=604;
temp= 2*a12*m3 + 2*a13*m2 + 2*a21*m3 + 2*a23*m1 + 2*a31*m2 + 2*a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=605;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=606;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=607;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=608;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=609;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=610;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=611;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=612;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=613;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=614;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=615;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=616;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=617;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=618;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=619;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=620;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=621;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=622;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=623;
temp= 2*a11*m2 + a12*m1 + a21*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=624;
temp= 2*a11*m3 + a13*m1 + a31*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=625;
temp= a12*m2 + a21*m2 + 2*a22*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=626;
temp= a12*m3 + a13*m2 + a21*m3 + a31*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=627;
temp= a12*m3 + a21*m3 + a23*m1 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=628;
temp= a13*m2 + a23*m1 + a31*m2 + a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=629;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=630;
temp= a13*m3 + a31*m3 + 2*a33*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=631;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=632;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=633;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=634;
temp= 2*a12*m3 + 2*a13*m2 + 2*a21*m3 + 2*a23*m1 + 2*a31*m2 + 2*a32*m1;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=635;
temp= 2*a22*m3 + a23*m2 + a32*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

i=636;
temp= a23*m3 + a32*m3 + 2*a33*m2;
atomicAdd(&mat[threadIdx.x & (4-1)][i],P*temp);

     }




     __syncthreads();     
    
     //check whether the calculation in the last step is correct or not
     //if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.y==0 && threadIdx.x*2<44)
     //{
     //   printf("mat[%d][%d] %f \n",threadIdx.y,threadIdx.x*2,mat[threadIdx.y][threadIdx.x*2]);
     //   printf("mat[%d][%d] %f \n",threadIdx.y,threadIdx.x*2+1,mat[threadIdx.y][threadIdx.x*2+1]);
     //}      
     //__syncthreads();  

     // copy the coefficient matrix in each block to the global memory

     if (M>=4)
     {
	     if (threadIdx.y<4)
	     {
	       double index1;
	       index1=blockIdx.y*gridDim.x*(4*n_c)+blockIdx.x*(4*n_c)+threadIdx.y*n_c+threadIdx.x*20;
	       index1=fmod(index1,4096.0*4*n_c);
	       int index;
	       index=__double2int_rn(index1);

	       for (int ii=0;ii<20;ii++)
	       {
		   atomicAdd((B_C+index+ii),mat[threadIdx.y][threadIdx.x*20+ii]);
	       }
	     }
     } 
     else
     {
        if (threadIdx.y==0)
        {
               for (int jj=0; jj<4; jj++)
               {
		       double index1;
		       index1=blockIdx.y*gridDim.x*(4*n_c)+blockIdx.x*(4*n_c)+jj*n_c+threadIdx.x*20;
		       index1=fmod(index1,4096.0*4*n_c);
		       int index;
		       index=__double2int_rn(index1);

		       for (int ii=0;ii<20;ii++)
		       {
			   atomicAdd((B_C+index+ii),mat[jj][threadIdx.x*20+ii]);
		       }
               }
        }
     }
                                                                                                                   
     __syncthreads(); 
      
}


