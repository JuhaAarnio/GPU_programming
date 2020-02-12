
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

const int blockSize = 32;
const int N = 32;
const int GALAXY_COUNT = 100000;

__global__ void calculateAngles(char* a, int* b) {
    
}

char* readFile(char* filename) {
    int i = 0;
    FILE* fp;
    char* coordinates{};
    char* galaxies[GALAXY_COUNT];
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Unable to open the file");
    }
    while (fgets(coordinates, GALAXY_COUNT, fp) != NULL) {
        galaxies[i] = coordinates;
        i++;
    }
    return galaxies;
}

int main() {
    char* realGalaxies = readFile("data_100k_arcmin.txt");
    char* syntheticGalaxies = readFile("flat_100k_arcmin.txt");
}