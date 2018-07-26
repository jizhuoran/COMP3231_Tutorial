//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>



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

  int nblocks  = 32, nthreads = 128;

  int N = nblocks * nthreads;

  std::vector<int> a(N), b(N), c(N);
  int *dev_a, *dev_b, *dev_c;

  for(int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));

  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  add<<<N,1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);


  for( int i = 0; i < N; i++ ){
    printf( "cpu: %d, gpu: %d\n", a[i]+b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  cudaDeviceReset();

  return 0;
}
