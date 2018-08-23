INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS := -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all: julia

julia: julia.cu Makefile
	nvcc julia.cu -o julia $(INC) $(NVCCFLAGS) $(LIB)


clean:
	rm -f julia.cu
