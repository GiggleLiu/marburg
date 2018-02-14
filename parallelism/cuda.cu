#include <iostream>
#include <sys/time.h>

__global__ void saxpyDevice(int n, float a, float *x, float *y){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = a*x[i] + y[i];
}

void saxpy(int n, float a, float *x, float *y){
    float *d_x, *d_y;

    // allocate GPU memory, and upload data
    cudaMalloc(&d_x, n*sizeof(float)); 
    cudaMalloc(&d_y, n*sizeof(float));
    cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);

    // send instructions to GPU
    saxpyDevice<<<(n+255)/256, 256>>>(n, 2.0f, d_x, d_y);

    // download data, and free GPU memory
    cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(void){
    int N = 1<<20;
    float *x, *y;
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int i=0; i<100; i++)
        saxpy(N, 2.0f, x, y);
    gettimeofday(&t1, NULL);
    std::cout<<"CUDA = "<<(t1.tv_sec - t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000<<"ms"<<std::endl;
	return 0;
}
