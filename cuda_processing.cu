#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS 256

__global__ void grayscaleKernel(
    unsigned char *input,
    unsigned char *output,
    int totalPixels
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < totalPixels) {
        int idx = i * 3;

        int gray = (299 * input[idx]
                   + 587 * input[idx + 1]
                   + 114 * input[idx + 2] + 500) / 1000;

        output[i] = (unsigned char)gray;
    }
}

__global__ void brightnessKernel(
    unsigned char *input,
    unsigned char *output,
    int size,
    int value
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        int result = input[i] + value;
        output[i] = max(0, min(255, result));
    }
}

__global__ void contrastKernel(
    unsigned char *input,
    unsigned char *output,
    int size,
    float factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        int result = (int)floorf((input[i] - 128) * factor + 128.0f + 0.5f);
        output[i] = max(0, min(255, result));
    }
}

__global__ void inversionKernel(
    unsigned char *input,
    unsigned char *output,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        output[i] = 255 - input[i];
    }
}

int main(int argc, char **argv) {

    if (argc != 3) {
        printf("Usage: ./cuda_processing input.bin image_count\n");
        return 1;
    }

    int imageCount = atoi(argv[2]);
    int totalPixels = imageCount * 32 * 32;
    int totalBytes = totalPixels * 3;

    unsigned char *h_input =
        (unsigned char*)malloc(totalBytes);

    unsigned char *h_gray =
        (unsigned char*)malloc(totalPixels);

    unsigned char *h_output =
        (unsigned char*)malloc(totalBytes);

    FILE *file = fopen(argv[1], "rb");

    if (!file || fread(h_input, 1, totalBytes, file) != totalBytes) {
        printf("Input file error\n");
        return 1;
    }

    fclose(file);

    unsigned char *d_input, *d_output, *d_gray;

    cudaMalloc(&d_input, totalBytes);
    cudaMalloc(&d_output, totalBytes);
    cudaMalloc(&d_gray, totalPixels);

    cudaMemcpy(d_input, h_input, totalBytes, cudaMemcpyHostToDevice);

    int blocks = (totalBytes + THREADS - 1) / THREADS;

    grayscaleKernel<<<(totalPixels + THREADS - 1) / THREADS, THREADS>>>(
        d_input, d_gray, totalPixels
    );

    brightnessKernel<<<blocks, THREADS>>>(
        d_input, d_output, totalBytes, 40
    );

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    FILE *out = fopen("output/gpu_brightness.bin", "wb");
    fwrite(h_output, 1, totalBytes, out);
    fclose(out);

    contrastKernel<<<blocks, THREADS>>>(
        d_input, d_output, totalBytes, 1.5f
    );

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    out = fopen("output/gpu_contrast.bin", "wb");
    fwrite(h_output, 1, totalBytes, out);
    fclose(out);

    inversionKernel<<<blocks, THREADS>>>(
        d_input, d_output, totalBytes
    );

    cudaMemcpy(h_output, d_output, totalBytes, cudaMemcpyDeviceToHost);

    out = fopen("output/gpu_inversion.bin", "wb");
    fwrite(h_output, 1, totalBytes, out);
    fclose(out);

    cudaMemcpy(h_gray, d_gray, totalPixels, cudaMemcpyDeviceToHost);

    out = fopen("output/gpu_grayscale.bin", "wb");
    fwrite(h_gray, 1, totalPixels, out);
    fclose(out);

    printf("CUDA processing completed!\n");
    printf("Images processed: %d\n", imageCount);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gray);

    free(h_input);
    free(h_output);
    free(h_gray);

    return 0;
}
