//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// #define BLOCK_NUM 16
// #define THREAD_NUM 64
#define N 1024*1024


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

__global__ void add(int *a, int *b, int *c, int *d, int *e, int *f, int *g) {

  int tid = blockIdx.x; // handle the data at this index
  
  int local[6];

  local[0] = a[tid];
  local[1] = b[tid];
  local[2] = c[tid];
  local[3] = d[tid];
  local[4] = e[tid];
  local[5] = f[tid];

  __syncthreads();

  if(tid < N) {
    int sum = 0;
    for (int i = 0; i < 6; ++i) {
      sum += local[i];
    }
    g[tid] = sum;
  }

}


//
// host code
//

int main(int argc, const char **argv) {



  int *a, *b, *c, *d, *e, *f, *g;
  int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e, *dev_f, *dev_g;


  a = (int*) malloc(sizeof(int) * N);
  b = (int*) malloc(sizeof(int) * N);
  c = (int*) malloc(sizeof(int) * N);
  d = (int*) malloc(sizeof(int) * N);
  e = (int*) malloc(sizeof(int) * N);
  f = (int*) malloc(sizeof(int) * N);
  g = (int*) malloc(sizeof(int) * N);


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


  add<<<1024,N/1024>>>(dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, dev_g);

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
