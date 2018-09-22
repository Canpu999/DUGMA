
#include <cuda_runtime.h>
#include <stdio.h>

void check(cudaError_t error_id)
{
  if (error_id != cudaSuccess) {
  printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id,cudaGetErrorString(error_id));
  printf("Result = FAIL\n");
  exit(EXIT_FAILURE);
  }
}

void fr_m(
        float *Ptr[6]
        )

{ 

  // free device memory
    // free temporal device global memory
  check(cudaFree(Ptr[0]));
  check(cudaFree(Ptr[1]));
  check(cudaFree(Ptr[2]));
  check(cudaFree(Ptr[3]));
  check(cudaFree(Ptr[4]));
  check(cudaFree(Ptr[5]));
  check(cudaDeviceReset());
  printf("Succeed in freeing the device memory \n");
  return;
}
