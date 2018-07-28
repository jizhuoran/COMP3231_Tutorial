#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void cuda_checker(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) (cuda_checker(err, __FILE__, __LINE__ ))



const int M = 1024, K = 1024, N = 1024;
const int TS = 32;

__global__ void myGEMM1(const float* A,
                        const float* B, 
                            float* C) {
    
    // Thread identifiers
    const int globalRow = threadIdx.x + blockIdx.x * blockDim.x;
    const int globalCol = threadIdx.y + blockIdx.y * blockDim.y;
 
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
 
    // Store the result
    C[globalCol*M + globalRow] = acc;
}




int main(int argc, const char **argv) {



  int a[N], b[N], c[N];
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

  add<<<(M / TS,N / TS), (TS, TS)>>>(dev_a, dev_b, dev_c);

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