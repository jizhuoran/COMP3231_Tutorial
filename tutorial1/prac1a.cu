#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// kernel code
// 

__global__ void my_first_kernel() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


//
// host code
//

int main(int argc, char **argv) {

  // set number of blocks, and threads per block

  int nblocks = 4, nthreads = 8; 

  
  // lanuch the kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
