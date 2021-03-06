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




#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__device__ float cache[threadsPerBlock * blocksPerGrid];

__global__ void dot(float* a, float* b, float* c) {
	
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = tid;
	
	float temp = 0;
	while (tid < N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	// set the cache values
	cache[cacheIndex] = temp;
	
	// synchronize threads in this block
	__syncthreads();
	
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i + blockIdx.x * blockDim.x)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex % blockDim.x == 0)
		c[blockIdx.x] = cache[blockIdx.x * blockDim.x];
}


int main (void) {
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	
	// allocate memory on the cpu side
	a = (float*)malloc(N*sizeof(float));
	b = (float*)malloc(N*sizeof(float));
	partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
	
	// allocate the memory on the gpu
	CUDA_CHECK(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_b, N*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&dev_partial_c, blocksPerGrid*sizeof(float)));
	
	// fill in the host mempory with data
	for(int i=0; i<N; i++) {
		a[i] = i;
		b[i] = i*2;
	}
	
	
	// copy the arrays 'a' and 'b' to the gpu
	CUDA_CHECK(cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice));
	
	
	float time;
	cudaEvent_t start, stop;

	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start, 0));


	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
	
	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));

	printf("Time to generate:  %3.1f ms \n", time);

	// copy the array 'c' back from the gpu to the cpu
	CUDA_CHECK(cudaMemcpy(partial_c,dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));
	
	// finish up on the cpu side
	c = 0;
	for(int i=0; i<blocksPerGrid; i++) {
		c += partial_c[i];
	}
	
	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));
	
	// free memory on the gpu side
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);
	
	// free memory on the cpu side
	free(a);
	free(b);
	free(partial_c);
}