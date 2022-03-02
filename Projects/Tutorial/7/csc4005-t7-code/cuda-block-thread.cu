#include <iostream>

__device__
int getBlockId() {
  return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__
int getLocalThreadId() {
  return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__
int getThreadId() {
  int blockId = getBlockId();
  int localThreadId = getLocalThreadId();
  return blockId * (blockDim.x * blockDim.y * blockDim.z) + localThreadId;
}

__global__
void cuda_setnum() {
  if (getLocalThreadId() != 0) return;
  
}

__global__
void cuda_getid() {
    int blockId = getBlockId();
    int localThreadId = getLocalThreadId();
    int threadId = getThreadId();
    printf("Hi. My blockId=%d, localThreadid=%d, threadId=%d\n", blockId, localThreadId, threadId);
}

int main() {
    dim3 grid_size;
    grid_size.x = 3;
    grid_size.y = 2;
    grid_size.z = 1;

    dim3 block_size;
    block_size.x = 5;
    block_size.y = 4;
    block_size.z = 1;

    cuda_getid<<<grid_size,block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}