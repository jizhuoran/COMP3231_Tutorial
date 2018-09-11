
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	test

test:	test.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc test.cu -o test $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f test
