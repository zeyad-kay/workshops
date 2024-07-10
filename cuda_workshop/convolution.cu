#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 32
#define INPUT_WIDTH 512   
#define INPUT_HEIGHT 512     
#define MASK_HEIGHT 3     
#define MASK_WIDTH 3
#define OUTPUT_WIDTH (INPUT_WIDTH - MASK_WIDTH + 1)
#define OUTPUT_HEIGHT (INPUT_HEIGHT - MASK_HEIGHT + 1)


__global__ void Convolution(float* input, float* mask, float* output, int numARows, int numACols, int numBRows, int numBCols, int numCRows, int numCCols)
{
	int col = blockIdx.x * (BLOCK_SIZE - MASK_WIDTH + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - MASK_WIDTH + 1) + threadIdx.y;
	int row_i = row - MASK_WIDTH + 1;
	int col_i = col - MASK_WIDTH + 1;

	float tmp = 0;

	__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < INPUT_WIDTH && row_i >= 0 && col_i < INPUT_WIDTH && col_i >= 0)
	{
		shm[threadIdx.y][threadIdx.x] = input[col_i * INPUT_WIDTH + row_i];
	}
	else
	{
		shm[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	if (threadIdx.y < (BLOCK_SIZE - MASK_WIDTH + 1) && threadIdx.x < (BLOCK_SIZE - MASK_WIDTH + 1) && row < (OUTPUT_WIDTH - MASK_WIDTH + 1) && col < (OUTPUT_WIDTH - MASK_WIDTH + 1))
	{
		for (int i = 0; i< MASK_WIDTH;i++)
			for (int j = 0;j<MASK_WIDTH;j++)
				tmp += shm[threadIdx.y + i][threadIdx.x + j] * output[j*MASK_WIDTH + i];
		mask[col*OUTPUT_WIDTH + row] = tmp;
	}
}


void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv)
{
	srand(1);
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	unsigned int size_input = INPUT_WIDTH * INPUT_HEIGHT;
	unsigned int mem_size_input = sizeof(float) * size_input;
	float* input = (float*)malloc(mem_size_input);

	unsigned int size_mask = OUTPUT_WIDTH * OUTPUT_HEIGHT;
	unsigned int mem_size_mask = sizeof(float) * size_mask;
	float* mask = (float*)malloc(mem_size_mask);

	unsigned int size_output = MASK_WIDTH * MASK_HEIGHT;
	unsigned int mem_size_output = sizeof(float) * size_output;
	float* output = (float*)malloc(mem_size_output);

	randomInit(input, size_input);
	randomInit(output, size_output);

	float* input_cuda;
	float* mask_cuda;
	float* output_cuda;

	cudaMalloc((void**)&input_cuda, mem_size_input);

	cudaMalloc((void**)&mask_cuda, mem_size_mask);

	cudaMalloc((void**)&output_cuda, mem_size_output);

	cudaMemcpy(input_cuda, input, mem_size_input, cudaMemcpyHostToDevice);

	cudaMemcpy(output_cuda, output, mem_size_output, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((OUTPUT_WIDTH - 1) / (BLOCK_SIZE - MASK_WIDTH + 1), (OUTPUT_WIDTH - 1) / (BLOCK_SIZE - MASK_WIDTH + 1));

	cudaEventRecord(start);

	Convolution <<< grid, threads >>>(input_cuda, mask_cuda, output_cuda, INPUT_HEIGHT, INPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_WIDTH, MASK_HEIGHT, MASK_WIDTH);

    cudaDeviceSynchronize();

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	cudaMemcpy(mask, mask_cuda, mem_size_mask, cudaMemcpyDeviceToHost);

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start, stop);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms\n", INPUT_WIDTH, INPUT_HEIGHT, miliseconds);

	free(input);
	free(mask);
	free(output);
	cudaFree(input_cuda);
	cudaFree(mask_cuda);
	cudaFree(output_cuda);

	return 0;
}