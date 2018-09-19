INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: julia blur

julia: julia.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc julia.cu -o julia $(INC) $(NVCCFLAGS) $(LIB)


blur: blur.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc blur.cu -o blur $(INC) $(NVCCFLAGS) $(LIB)


clean:
	rm -f julia blur
