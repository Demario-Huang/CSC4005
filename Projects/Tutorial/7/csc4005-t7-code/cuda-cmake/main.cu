#include <iostream>
__shared__ int t;
__global__ void cuda_hello(){
    printf("Hello World from GPU! %d\n", t);
    t = 1;
    printf("Hello World from GPU! %d\n", t);
}

int main() {
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
