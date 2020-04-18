
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <math.h>
using namespace std;

const int blockSize = 32;
const int GALAXY_COUNT = 200000;
const float pi = 3.141592653589;
const float conversionFactor = (float)1 / (float)60 * pi / (float)180;

__global__ void calculateAngles(float* d_realGalaxies, float* d_syntheticGalaxies, float* d_dotProducts) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int z = 0;
	int y = 0;
	for (int i = index; i < 10000000000; i += stride)
	{
		for (int j = 0; j < 100000; j = j + 2) {
			d_dotProducts[z] = d_realGalaxies[y] * d_realGalaxies[j + 1] + d_realGalaxies[y + 1] * d_realGalaxies[j];
			if (d_dotProducts[z] < -1) {
				d_dotProducts[z] == -1;
			}
			if (d_dotProducts[z] > 1) {
				d_dotProducts[z] == 1;
			}
			++z;
			y = y + 2;
		}
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
		// Converting from arch minutes to radians.
		galaxies[i] = coordinates * conversionFactor;
		++i;
	}
	file.close();
	return galaxies;
}

int main() {
	printf("%f", conversionFactor);
	int arraySize = (GALAXY_COUNT / 2) * sizeof(float);
	float* galaxyAngles = new float[arraySize * arraySize];
	float* h_realGalaxies = readFile("data_100k_arcmin.txt");
	float* h_syntheticGalaxies = readFile("flat_100k_arcmin.txt");
	float* h_dotProducts = new float[arraySize * arraySize];
	float* d_realGalaxies; cudaMalloc(&d_realGalaxies, arraySize);
	float* d_syntheticGalaxies; cudaMalloc(&d_syntheticGalaxies, arraySize);
	float* d_dotProducts; cudaMalloc(&d_dotProducts, arraySize);
	// Intializing the CUDA computation kernel
	int threadsInBlock = 256;
	int blocksInGrid = 100;
	cudaMemcpy(d_realGalaxies, h_realGalaxies, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_syntheticGalaxies, h_syntheticGalaxies, arraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dotProducts, h_dotProducts, arraySize, cudaMemcpyHostToDevice);
	// Execute the function on GPU
	calculateAngles <<<blocksInGrid, threadsInBlock>>> (d_realGalaxies, d_syntheticGalaxies, d_dotProducts);
	cudaMemcpy(h_dotProducts, d_dotProducts, arraySize, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 1000000; i++)
	{
		galaxyAngles[i] = h_dotProducts[i];
	}
	for (int i = 0; i < 1000000; i++)
	{
		printf("%f \n", galaxyAngles[i]);
	}
}