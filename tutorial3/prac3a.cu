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



const int M = 16, K = 16, N = 16;
const int TS = 4;


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
    C[globalCol*M + globalRow] = 1;
}




int main(int argc, const char **argv) {



  float a[M*N], b[M*N], c[M*N];
  float *dev_a, *dev_b, *dev_c;

  printf( "come to here!!!");

  for(int i = 0; i < M*N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  printf( "come to here!!!");

  CUDA_CHECK( cudaMalloc((void**)&dev_a, M * N * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_b, M * N  * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_c, M * N  * sizeof(float)) );

  CUDA_CHECK( cudaMemcpy(dev_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_b, b, M * N * sizeof(float), cudaMemcpyHostToDevice) );

  myGEMM1<<<(M / TS,N / TS), (TS, TS)>>>(dev_a, dev_b, dev_c);

  CUDA_CHECK( cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost) );


  for( int i = 0; i < M*N; i++ ){
    printf( "cpu: %f, gpu: %f\n", c[i], c[i]);
  }

  CUDA_CHECK( cudaFree(dev_a) );
  CUDA_CHECK( cudaFree(dev_b) );
  CUDA_CHECK( cudaFree(dev_c) );

  cudaDeviceReset();

  return 0;
}