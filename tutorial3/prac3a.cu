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



const int M = 2048, K = 2048, N = 2048;
const int TS = 32;


__global__ void myGEMM1(const float* A,
                        const float* B, 
                              float* C) {
    
    // Thread identifiers
    const int globalRow = threadIdx.x + blockIdx.x * blockDim.x;
    const int globalCol = threadIdx.y + blockIdx.y * blockDim.y;
    

    // printf("Hello from block %d, thread %d\n", globalRow, globalCol);


    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + globalRow] * B[globalCol*K + k];
    }
 
    // Store the result
    C[globalCol*M + globalRow] = acc;
}




int main(int argc, const char **argv) {



  float* a,b,c;
  a = (float*) malloc(sizeof(float) * M * K);  
  b = (float*) malloc(sizeof(float) * K * N);  
  c = (float*) malloc(sizeof(float) * M * N);  
  float *dev_a, *dev_b, *dev_c;

  printf( "come to here!!!");

  for(int i = 0; i < M*N; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  printf( "come to here!!!");

  CUDA_CHECK( cudaMalloc((void**)&dev_a, M * N * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_b, M * N  * sizeof(float)) );
  CUDA_CHECK( cudaMalloc((void**)&dev_c, M * N  * sizeof(float)) );

  CUDA_CHECK( cudaMemcpy(dev_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(dev_b, b, M * N * sizeof(float), cudaMemcpyHostToDevice) );

  dim3 threadsPerBlock(TS, TS);
  dim3 numBlocks(M / TS, N / TS);



  float time;
  cudaEvent_t start, stop;

  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));


  myGEMM1<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
  
  CUDA_CHECK( cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost) );

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

  printf("Time to generate:  %3.1f ms \n", time);


  CUDA_CHECK( cudaFree(dev_a) );
  CUDA_CHECK( cudaFree(dev_b) );
  CUDA_CHECK( cudaFree(dev_c) );

  cudaDeviceReset();

  return 0;
}