ALL=avx2 cpu cuda

CPU=g++
CPU_FLAGS=-O3

AVX2=g++
AVX2_FLAGS=-mavx -mfma -O3

CUDA=nvcc
CUDA_FLAGS=-arch=sm_30

all: $(ALL)

avx2: avx2.cpp
	$(AVX2) $(AVX2_FLAGS) $< -o $@

cuda: cuda.cu
	$(CUDA) $(CUDA_FLAGS) $< -o $@

cpu: cpu.cpp
	$(CPU) $(CPU_FLAGS) $< -o $@

clean:
	rm $(ALL)
