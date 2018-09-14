
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	prac1a prac1b prac1c

prac1a:	prac1a.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac1a.cu -o prac1a $(INC) $(NVCCFLAGS) $(LIB)

prac1b:	prac1b.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac1b.cu -o prac1b $(INC) $(NVCCFLAGS) $(LIB)

prac1c:	prac1c.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac1c.cu -o prac1c $(INC) $(NVCCFLAGS) $(LIB)

prac1d:	prac1d.cu Makefile
	/usr/local/cuda-9.2/bin/nvcc prac1d.cu -o prac1d $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f prac1a prac1b prac1c prac1d
