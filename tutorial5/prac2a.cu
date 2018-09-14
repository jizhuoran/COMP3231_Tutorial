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



  int a[N], b[N], c[N], d[N], e[N], f[N], g[N];
  int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e, *dev_f, *dev_g;

  for(int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = -i;
    c[i] = -i;
    d[i] = i * i;
    e[i] = i * i;
    f[i] = i * i;
  }

  CUDA_CHECK( cudaMalloc((void**)&dev_a, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_b, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_c, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_d, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_e, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_f, N * sizeof(int)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_g, N * sizeof(int)) );

  CUDA_CHECK( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_c, c, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_d, d, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_e, e, N * sizeof(int), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_f, f, N * sizeof(int), cudaMemcpyHostToDevice) );


  float time;
  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));


  add<<<N,1>>>(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f);

  CUDA_CHECK( cudaMemcpy(g, dev_g, N * sizeof(int), cudaMemcpyDeviceToHost) );

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

  printf("Time to generate:  %3.1f ms \n", time);

  // for( int i = 0; i < N; i++ ){
    // printf( "cpu: %d, gpu: %d\n", a[i]+b[i], c[i]);
  // }

  CUDA_CHECK( cudaFree(dev_a) );
  CUDA_CHECK( cudaFree(dev_b) );
  CUDA_CHECK( cudaFree(dev_c) );
  CUDA_CHECK( cudaFree(dev_d) );
  CUDA_CHECK( cudaFree(dev_e) );
  CUDA_CHECK( cudaFree(dev_f) );
  CUDA_CHECK( cudaFree(dev_g) );

  cudaDeviceReset();

  return 0;
}
