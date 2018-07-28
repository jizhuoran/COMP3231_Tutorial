
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac4a prac4b prac4c

prac4a:	prac4a.cu Makefile
	nvcc prac4a.cu -o prac4a $(INC) $(NVCCFLAGS) $(LIB)

prac4b:	prac4b.cu Makefile
	nvcc prac4b.cu -o prac4b $(INC) $(NVCCFLAGS) $(LIB)

prac4c:	prac4c.cu Makefile
	nvcc prac4c.cu -o prac4c $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac4a prac4b prac4c
