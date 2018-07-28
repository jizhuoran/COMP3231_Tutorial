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



const int M = 512, K = 512, N = 512;
const int TS = 32;


__global__ void myGEMM2(const float* A,
                        const float* B, 
                              float* C) {
    
    // Thread identifiers
    const int row = threadIdx.x; // Local row ID (max: TS)
    const int col = threadIdx.y; // Local col ID (max: TS)
    const int globalRow = row + blockIdx.x * blockDim.x;
    const int globalCol = col + blockIdx.y * blockDim.y;
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __shared__ float Asub[TS][TS];
    __shared__ float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        __syncthreads();
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        __syncthreads();
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}




int main(int argc, const char **argv) {



  float a[M*N], b[M*N], c[M*N];
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


  myGEMM2<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);
  
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