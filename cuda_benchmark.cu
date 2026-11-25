#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256
#define ITERATIONS 10

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void grayscaleKernel(unsigned char *in, unsigned char *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = i * 3;
        out[i] = (299 * in[j] + 587 * in[j+1] +
                  114 * in[j+2] + 500) / 1000;
    }
}

__global__ void brightnessKernel(unsigned char *in, unsigned char *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = in[i] + 40;
        out[i] = max(0, min(255, v));
    }
}

__global__ void contrastKernel(unsigned char *in, unsigned char *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = (int)floorf((in[i] - 128) * 1.5f + 128.0f + 0.5f);
        out[i] = max(0, min(255, v));
    }
}

__global__ void inversionKernel(unsigned char *in, unsigned char *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 255 - in[i];
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: ./cuda_benchmark input.bin image_count\n");
        return 1;
    }

    int count = atoi(argv[2]);
    int pixels = count * 32 * 32;
    int bytes = pixels * 3;

    unsigned char *h_in = (unsigned char*)malloc(bytes);
    unsigned char *h_out = (unsigned char*)malloc(bytes);
    unsigned char *h_gray = (unsigned char*)malloc(pixels);

    if (!h_in || !h_out || !h_gray) return 1;

    FILE *f = fopen(argv[1], "rb");
    if (!f || fread(h_in, 1, bytes, f) != bytes) {
        printf("Input file error\n");
        return 1;
    }
    fclose(f);

    unsigned char *d_in, *d_out, *d_gray;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMalloc(&d_gray, pixels));

    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int bb = (bytes + THREADS - 1) / THREADS;
    int bp = (pixels + THREADS - 1) / THREADS;

    printf("Operation,Kernel_ms,End_to_end_ms\n");

    for (int op = 0; op < 4; op++) {
        float kernelTotal = 0, totalTotal = 0;

        for (int r = 0; r < ITERATIONS + 2; r++) {
            CHECK(cudaEventRecord(start));

            if (op == 0)
                grayscaleKernel<<<bp, THREADS>>>(d_in, d_gray, pixels);
            else if (op == 1)
                brightnessKernel<<<bb, THREADS>>>(d_in, d_out, bytes);
            else if (op == 2)
                contrastKernel<<<bb, THREADS>>>(d_in, d_out, bytes);
            else
                inversionKernel<<<bb, THREADS>>>(d_in, d_out, bytes);

            CHECK(cudaGetLastError());
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));

            float kt;
            CHECK(cudaEventElapsedTime(&kt, start, stop));

            CHECK(cudaEventRecord(start));
            CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

            if (op == 0) {
                grayscaleKernel<<<bp, THREADS>>>(d_in, d_gray, pixels);
                CHECK(cudaMemcpy(h_gray, d_gray, pixels, cudaMemcpyDeviceToHost));
            } else {
                if (op == 1)
                    brightnessKernel<<<bb, THREADS>>>(d_in, d_out, bytes);
                else if (op == 2)
                    contrastKernel<<<bb, THREADS>>>(d_in, d_out, bytes);
                else
                    inversionKernel<<<bb, THREADS>>>(d_in, d_out, bytes);

                CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
            }

            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));

            float tt;
            CHECK(cudaEventElapsedTime(&tt, start, stop));

            if (r >= 2) {
                kernelTotal += kt;
                totalTotal += tt;
            }
        }

        const char *names[] = {"grayscale", "brightness", "contrast", "inversion"};
        printf("%s,%.6f,%.6f\n", names[op],
               kernelTotal / ITERATIONS, totalTotal / ITERATIONS);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gray);
    free(h_in);
    free(h_out);
    free(h_gray);

    return 0;
}
