# A Simple Example of C level Acceleration Using Parallelism
To start,
```bash
$ make
$ ./cpu
$ ./avx2
$ ./cuda
```
It calculates saxpy function, and print system time elapse.

1. Realization without parallelism: *cpu.cpp*. Here, you should not use `-O3` tag during compilation, otherwise, g++ uses avx2 automatically. Notice this automatic optimization is only achievable for simple functions.
2. CPU parallelism using AVX2 instruction set: *avx2.cpp*.
3. GPU parallelism using CUDA programming model: *cuda.cu*.
It requires a CUDA library, and compiles using `nvcc`.
Here, you will not see a GPU acceleration!
Because the data transfer between system memory and GPU memory has a lot overhead and the complexity of saxpy function is only $O(N)$.
To confirm this, time the excution part of program only please, you will see an amazing acceleration.
