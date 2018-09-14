
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac1a prac1b prac1c

prac2a:	prac1a.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac1a.cu -o prac1a $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac1a
