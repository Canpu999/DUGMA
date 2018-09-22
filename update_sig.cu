
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>



#define MAX_SIZE_DATA 3 
#define MAX_SIZE_TRAN 12
#define times2D 1      // the coefficient matrix in shared memory should be 44*32*times2D
#define times3D 2      // the coefficient matrix in shared memory should be 4*637*times3D   
#define scale_accuracy 1000000;
 
 __constant__ int d_Data_Coef[MAX_SIZE_DATA]; //include N,M,D
 __constant__ double d_Tran_Coef[MAX_SIZE_TRAN];


void check(cudaError_t error_id)
{
  if (error_id != cudaSuccess) {
  printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id,cudaGetErrorString(error_id));
  printf("Result = FAIL\n");
  exit(EXIT_FAILURE);
  }
}


// copy M,N,D,R,t to constant memory
void setup_coef_constant(int N, int M, int D, double *R, double *t)
{
  const int h_Data_Coef[MAX_SIZE_DATA]={N,M,D};
  // copy data on host to the constant memory on device
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


__global__ void init_distX(uint *distance_X)
{
    int ix;
    ix=blockIdx.x*blockDim.x+threadIdx.x;
    int N;
    N=d_Data_Coef[0];
    if (ix<N) *(distance_X+ix)=4000000000;
}

__global__ void convert_back(uint *distance_X,float * distance_XX)
{
    int ix;
    ix=blockIdx.x*blockDim.x+threadIdx.x;
    int N;
    N=d_Data_Coef[0];
    float ss;
    if (ix<N) 
    {
        ss=__uint2float_rn(*(distance_X+ix));
	*(distance_XX+ix)=ss/scale_accuracy;
    }
}

__global__ void init_distY(uint *distance_Y)
{
    int ix;
    ix=blockIdx.x*blockDim.x+threadIdx.x;
    int M;
    M=d_Data_Coef[1];
    if (ix<M) *(distance_Y+ix)=4000000000;
}

__global__ void compi_distance2D(float *d_X,float *d_Y,  uint *distance_X,uint *distance_Y)
{
      int ix,iy,N,M,D;
      ix=blockIdx.x*blockDim.x+threadIdx.x;
      iy=blockIdx.y*blockDim.y+threadIdx.y;
      N=d_Data_Coef[0];
      M=d_Data_Coef[1];
      D=d_Data_Coef[2];
      double r11,r21,r12,r22,t1,t2; // the old R,t
      r11=d_Tran_Coef[0];
      r21=d_Tran_Coef[1];
      r12=d_Tran_Coef[2];
      r22=d_Tran_Coef[3];
      t1=d_Tran_Coef[4];
      t2=d_Tran_Coef[5]; 

      __shared__ float X[64],Y[64];

      // put the data related to x,y into the shared memory
      if ((threadIdx.y==0) && (ix<N))
      {
         X[threadIdx.x*D]=*(d_X+ix*D);

         X[threadIdx.x*D+1]=*(d_X+ix*D+1);
 
      }

      if ((threadIdx.y==0)  && (blockIdx.y*blockDim.y+threadIdx.x<M))
      {
         Y[threadIdx.x*D]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D,Y[threadIdx.x*D]);         
         Y[threadIdx.x*D+1]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+1);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D+1,Y[threadIdx.x*D+1]); 

      }             
      __syncthreads();
      
      __shared__ uint dist_X[32],dist_Y[32]; 
      __shared__ float dist;

     // initialize all the elements in mat to zero
     dist_X[threadIdx.x]=4000000000;
     dist_Y[threadIdx.x]=4000000000;
                                                                                                
     __syncthreads();
     
    
     //calculate the minimum distance: x to every point in Y

     if (ix<N && iy<M)
     {
       float n1,n2,m1,m2,dd1,dd2;
       n1=X[2*threadIdx.x];
       n2=X[2*threadIdx.x+1];
       m1=Y[2*threadIdx.y];
       m2=Y[2*threadIdx.y+1];
       dd1=r11*m1+r12*m2+t1;       
       dd2=r21*m1+r22*m2+t2;
       m1=dd1;
       m2=dd2;       
       float s;
       float DD=D;
       unsigned int ss;
       s=((n1-m1)*(n1-m1)+(n2-m2)*(n2-m2))/DD*scale_accuracy;
       // avoid the race condition
       ss=__float2uint_rn(s);
       atomicMin(&dist_X[threadIdx.x],ss);
     }

     __syncthreads();  
     if ((ix<N) && (iy<M) && (threadIdx.y==0))
     {
        atomicMin(distance_X+ix,dist_X[threadIdx.x]);        
     }   
     __syncthreads(); 
         
}


__global__ void compi_distance3D(float *d_X,float *d_Y,  uint *distance_X,uint *distance_Y)
{
      int ix,iy,N,M,D;
      ix=blockIdx.x*blockDim.x+threadIdx.x;
      iy=blockIdx.y*blockDim.y+threadIdx.y;
      N=d_Data_Coef[0];
      M=d_Data_Coef[1];
      D=d_Data_Coef[2];
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

      __shared__ float X[96],Y[96];

      // put the data related to x,y into the shared memory
      if ((threadIdx.y==0) && (ix<N))
      {
         X[threadIdx.x*D]=*(d_X+ix*D);

         X[threadIdx.x*D+1]=*(d_X+ix*D+1);

         X[threadIdx.x*D+2]=*(d_X+ix*D+2); 
      }

      if ((threadIdx.y==0)  && (blockIdx.y*blockDim.y+threadIdx.x<M))
      {
         Y[threadIdx.x*D]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D,Y[threadIdx.x*D]);         
         Y[threadIdx.x*D+1]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+1);
         //printf("Y[%d]= %f \n",blockIdx.y*blockDim.y*D+threadIdx.x*D+1,Y[threadIdx.x*D+1]); 
         Y[threadIdx.x*D+2]=*(d_Y+blockIdx.y*blockDim.y*D+threadIdx.x*D+2);

      }             
      __syncthreads();
      
      __shared__ uint dist_X[32],dist_Y[32]; 
      __shared__ float dist;

     // initialize all the elements in mat to zero
     dist_X[threadIdx.x]=4000000000;
     dist_Y[threadIdx.x]=4000000000;
                                                                                                
     __syncthreads();
     
    
     //calculate the minimum distance: x to every point in Y

     if (ix<N && iy<M)
     {
       float n1,n2,n3,m1,m2,m3,dd1,dd2,dd3;
       n1=X[3*threadIdx.x];
       n2=X[3*threadIdx.x+1];
       n3=X[3*threadIdx.x+2];
       m1=Y[3*threadIdx.y];
       m2=Y[3*threadIdx.y+1];
       m3=Y[3*threadIdx.y+2];
       dd1=r11*m1+r12*m2+r13*m3+t1;       
       dd2=r21*m1+r22*m2+r23*m3+t2;
       dd3=r31*m1+r32*m2+r33*m3+t3;
       m1=dd1;
       m2=dd2;       
       m3=dd3;
       float s;
       float DD=D;
       unsigned int ss;
       s=((n1-m1)*(n1-m1)+(n2-m2)*(n2-m2)+(n3-m3)*(n3-m3))/DD*scale_accuracy;
       // avoid the race condition
       ss=__float2uint_rn(s);
       atomicMin(&dist_X[threadIdx.x],ss);
     }

     __syncthreads();  
     if ((ix<N) && (iy<M) && (threadIdx.y==0))
     {
        atomicMin(distance_X+ix,dist_X[threadIdx.x]);        
     }   
     __syncthreads(); 
         
}


__global__ void compi_redu(float * B_C,int n_c, int L,int stride)
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


// d_idata is the array waiting to sum
// coefficient is the final sum whose number is n_c
// size is the length of the array
void Redu(float *d_idata,float *coefficient,int size, int n_c)
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
    int bytes=n_c*sizeof(float);
    cudaMemcpy(coefficient,d_idata,bytes,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    //for (int i=0; i<n_c; i++)
    //{
    //   printf(" coefficient[%d] = %f \n",i,*(coefficient+i));
    //}
}



void cpd(
        float *Ptr[6],double *R, double *t, double * N1, double * M1, double * D1,float  sig[1]
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


  setup_coef_constant(N, M, D, R,t);
  
  if (M>3000000 || N>3000000)
  {
     printf("The Number of the Points Exceeds the Maximum Threshold");
     exit(EXIT_FAILURE);
  }
  
  
  int NN,MM;  // the number of coefficient blocks in x, y direction
  unsigned int nBytes_X, nBytes_Y;

  NN=(N+32-1)/(32);  //blocks in X direction
  MM=(M+32-1)/(32);  //blocks in Y direction

  nBytes_X=N*sizeof(uint);
  nBytes_Y=M*sizeof(uint);
  
  uint *distance_X,*distance_Y;
  check(cudaMalloc((void **)&distance_X, nBytes_X));
  check(cudaMalloc((void **)&distance_Y, nBytes_Y));

  init_distX<<<NN,32>>>(distance_X);
  init_distY<<<MM,32>>>(distance_Y);
  check(cudaDeviceSynchronize());

  dim3 block(32,32,1);
  dim3 grid((N+block.x-1)/block.x, (M+block.y-1)/block.y);
  check(cudaDeviceSynchronize());

  // caculate the minimum distance
  if (D==2)
     compi_distance2D<<<grid, block>>>(d_X,d_Y,distance_X,distance_Y);
  else
     compi_distance3D<<<grid, block>>>(d_X,d_Y,distance_X,distance_Y);     
  check(cudaDeviceSynchronize());
  nBytes_X=N*sizeof(float);  
  float *distance_XX;
  check(cudaMalloc((void **)&distance_XX, nBytes_X));
  convert_back<<<NN,32>>>(distance_X,distance_XX);  
  check(cudaDeviceSynchronize());

  // calculate the sum
  float *sig1;
  sig1=(float*) malloc(4);
  Redu(distance_XX,sig1, N,1);
  sig[0]=(*sig1)/N;

  check(cudaDeviceSynchronize());

  free(sig1);
  check(cudaFree(distance_XX));
  check(cudaFree(distance_X));
  check(cudaFree(distance_Y));
  return;
}
