
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <vector>
using namespace std;

const int blockSize = 32;
const int GALAXY_COUNT = 200000;

__global__ void calculateAngles(float* d_realGalaxies, float* d_syntheticGalaxies, float* d_dotProducts) {
    float dotProduct = 0;
    for (int i = 0; i < 100000; i++)
    {
        dotProduct = d_realGalaxies[i] * d_syntheticGalaxies[i];
        d_dotProducts[i] = dotProduct;
    }
}

float* readFile(char* filename) {
    ifstream file;
    int i = 0;
    float coordinates;
    float* galaxies = new float[GALAXY_COUNT];
    file.open(filename);
    if (!file) {
        printf("Error opening the file");
        exit(1);
    }
    while (file >> coordinates)
    {
        galaxies[i] = coordinates;
        ++i;
    }
    file.close();
    return galaxies;
}

int main() {
    int arraySize = (GALAXY_COUNT / 2) * sizeof(float);
    float* h_realGalaxies = readFile("data_100k_arcmin.txt");
    float* h_syntheticGalaxies = readFile("flat_100k_arcmin.txt");
    float* h_dotProducts = new float[arraySize];
    float* d_realGalaxies; cudaMalloc(&d_realGalaxies, arraySize);
    float* d_syntheticGalaxies; cudaMalloc(&d_syntheticGalaxies, arraySize);
    float* d_dotProducts; cudaMalloc(&d_dotProducts, arraySize);
    // Intializing the CUDA computation kernel
    dim3 blockDimension(blockSize, 1);
    dim3 gridDimension(1, 1);
    cudaMemcpy(d_realGalaxies, h_realGalaxies, 100000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syntheticGalaxies, h_syntheticGalaxies, 100000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dotProducts, h_dotProducts, 100000 * sizeof(float), cudaMemcpyHostToDevice);
    // Execute the function on GPU
    calculateAngles<<<blockDimension, gridDimension>>>(d_realGalaxies, d_syntheticGalaxies, d_dotProducts);
    cudaMemcpy(h_dotProducts, d_dotProducts, arraySize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100000; i++)
    {
        printf("%f \n", h_dotProducts[i]);
    }
}