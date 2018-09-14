#include <stdio.h>

//
// kernel code
// 

__global__ void my_first_kernel() {
  printf("Hello from block (%d, %d), thread (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}


//
// host code
//

int main(int argc, char **argv) {

  // set number of blocks, and threads per block

  dim3 blocks_2d = dim3(2, 2);
  dim3 threads_2d = dim3(3, 3);


  
  // lanuch the kernel

  my_first_kernel<<<blocks_2d,threads_2d>>>();

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
