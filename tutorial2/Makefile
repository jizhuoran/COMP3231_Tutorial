
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac2a prac2b

prac2a:	prac2a.cu Makefile
	nvcc prac2a.cu -o prac2a $(INC) $(NVCCFLAGS) $(LIB)

prac2b:	prac2b.cu Makefile
	nvcc prac2b.cu -o prac2b $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac2a prac2b
