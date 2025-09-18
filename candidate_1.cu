#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

const int N = 1<<20;
const int THREADS = 256;

__global__ void vecAdd(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *hA, *hB, *hC, *hC_ref;
    cudaMallocHost(&hA, bytes);
    cudaMallocHost(&hB, bytes);
    cudaMallocHost(&hC, bytes);
    cudaMallocHost(&hC_ref, bytes);

    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(2*i);
        hC_ref[i] = hA[i] + hB[i];
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS - 1) / THREADS;

    vecAdd<<<blocks, THREADS>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    const int RUNS = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int r = 0; r < RUNS; ++r) {
        vecAdd<<<blocks, THREADS>>>(dA, dB, dC, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= RUNS;

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    const float tol = 1e-5f;
    for (int i = 0; i < N; ++i) {
        if (fabs(hC[i] - hC_ref[i]) > tol) {
            ok = false;
            break;
        }
    }

    printf("%s\n", ok ? "OK" : "FAIL");
    printf("TIME_MS %.3f\n", ms);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFreeHost(hC_ref);
    return 0;
}