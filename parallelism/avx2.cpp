#include <iostream>
#include <sys/time.h>
#include <mm_malloc.h>
#include <immintrin.h>

void saxpy(int n, float a, float *x, float *y){
	__m256 x_vec, y_vec, a_vec, res_vec;      //define the registers used
    a_vec = _mm256_set1_ps(a); /* Vector of 8 alpha values */
    for (int i=0; i<n; i+=8) {
        x_vec = _mm256_loadu_ps(&x[i]); /* Load 8 values from X */
        y_vec = _mm256_loadu_ps(&y[i]); /* Load 8 values from Y */
        res_vec = _mm256_fmadd_ps(a_vec, x_vec, y_vec); /* Compute */
        _mm256_store_ps(&y[i], res_vec);
    }
}

int main(){
    int N = 1<<20;
    float *x, *y;
    struct timeval t0, t1;

    x = (float*)_mm_malloc(N*sizeof(float), 32);
    y = (float*)_mm_malloc(N*sizeof(float), 32);

    gettimeofday(&t0, NULL);
    for (int i=0; i<100; i++)
        saxpy(N, 2.0f, x, y);
    gettimeofday(&t1, NULL);
    std::cout<<"AVX2 = "<<(t1.tv_sec - t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000<<"ms"<<std::endl;
	return 0;
}
