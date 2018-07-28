
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac3a prac3b prac3c

prac3a:	prac3a.cu Makefile
	nvcc prac3a.cu -o prac3a $(INC) $(NVCCFLAGS) $(LIB)

prac3b:	prac3b.cu Makefile
	nvcc prac3b.cu -o prac3b $(INC) $(NVCCFLAGS) $(LIB)

prac3c:	prac3c.cu Makefile
	nvcc prac3c.cu -o prac3c $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac3a prac3b prac3c
