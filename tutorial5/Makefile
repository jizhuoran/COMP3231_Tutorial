
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac2a

prac2a:	prac2a.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac2a.cu -o prac2a $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2a
