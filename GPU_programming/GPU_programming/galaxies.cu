
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
using namespace std;

const int blockSize = 32;
const int N = 32;
const int GALAXY_COUNT = 200000;

__global__ void calculateAngles(char* a, int* b) {
    float* d_realGalaxies; cudaMalloc(&d_realGalaxies, 100000 * sizeof(float));
    float* d_syntheticGalaxies; cudaMalloc(&d_syntheticGalaxies, 100000 * sizeof(float));
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
    float* h_realGalaxies = readFile("data_100k_arcmin.txt");
    float* h_syntheticGalaxies = readFile("flat_100k_arcmin.txt");
}