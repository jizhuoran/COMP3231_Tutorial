//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BLOCK_NUM 16
#define THREAD_NUM 64
#define N (BLOCK_NUM * THREAD_NUM)


static void cuda_checker(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) (cuda_checker(err, __FILE__, __LINE__ ))


//
// kernel code
// 

__global__ void add(int *a, int *b, int *c) {

  int tid = blockIdx.x; // handle the data at this index
  
  if(tid < N) {
    c[tid] = a[tid] + b[tid];
  }

}


//
// host code
//

int main(int argc, const char **argv) {



  int a[N], b[N], c[N], d[N];
  int *dev_a, *dev_b, *dev_c;

  for(int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  CUDA_CHECK( cudaMalloc((void**)&dev_a, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_b, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_c, N * sizeof(int)) );

  CUDA_CHECK( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) );

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  CUDA_CHECK( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) );


  for( int i = 0; i < N; i++ ){
    printf( "cpu: %d, gpu: %d\n", a[i]+b[i], c[i]);
  }

  CUDA_CHECK( cudaFree(dev_a) );
  CUDA_CHECK( cudaFree(dev_b) );
  CUDA_CHECK( cudaFree(dev_c) );

  cudaDeviceReset();

  return 0;
}
