//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define BLOCK_NUM (1024 * 32)
#define THREAD_NUM 32
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

  int tid = threadIdx.x + blockIdx.x * blockDim.x; // handle the data at this index
  
  if (tid < N) {
    if(tid % 2 == 0) {
      c[tid] = a[tid] + b[tid];
    } else {
      c[tid] = a[tid] - b[tid];
    }
  }
  

}


//
// host code
//

int main(int argc, const char **argv) {



  int *a, *b, *c;
  a = (int*) malloc(sizeof(int) * N);  
  b = (int*) malloc(sizeof(int) * N);  
  c = (int*) malloc(sizeof(int) * N);  
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


  float time;
  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));


  add<<<BLOCK_NUM, THREAD_NUM>>>(dev_a, dev_b, dev_c);

  CUDA_CHECK( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost) );

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

  printf("Time to generate:  %3.1f ms \n", time);

  // for( int i = 0; i < N; i++ ){
  //   int cpu_value = 0;

  //   if (i % 2 == 0) {
  //     a[i]+b[i];
  //   } else {
  //     a[i]-b[i];
  //   }

  //   printf( "cpu: %d, gpu: %d\n", cpu_value, c[i]);
  // }

  CUDA_CHECK( cudaFree(dev_a) );
  CUDA_CHECK( cudaFree(dev_b) );
  CUDA_CHECK( cudaFree(dev_c) );

  cudaDeviceReset();

  return 0;
}