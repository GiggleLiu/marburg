#include <cstdlib>
#include <iostream>
#include <sys/time.h>

void saxpy(int n, float a, float *x, float *y) {
    for (int i=0; i<n; i++)
        y[i] = a*x[i] + y[i];
}

int main(){
    int N = 1<<20;
    float *x, *y;
    struct timeval t0, t1;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    gettimeofday(&t0, NULL);
    for (int i=0; i<100; i++)
        saxpy(N, 2.0f, x, y);
    gettimeofday(&t1, NULL);
    std::cout<<"Base = "<<(t1.tv_sec - t0.tv_sec)*1000 + (t1.tv_usec-t0.tv_usec)/1000<<"ms"<<std::endl;
	return 0;
}
