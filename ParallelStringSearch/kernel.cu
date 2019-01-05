#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

__global__ void parallelGrep(char *global_data, int globalData_Size, char *key, int key_size, int *key_indexes, int *curr_index)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int key0 = key[0];

	if (idx < globalData_Size)
	{
		if (global_data[idx] == key0)
		{
			int save = 1;

			for (int i = 1; i < key_size; i++)
			{
				if (global_data[idx + i] != key[i])
				{
					save = 0;
					break;
				}
			}
			if (save == 1)
			{
				key_indexes[atomicAdd(curr_index, 1)] = idx;
			}
		}
	}

}

int main()
{
	FILE *pFile;
	size_t numB;
	long lSize;
	size_t key_size, key_ind_size;
	char *data_d, *data_h, *key_d;
	char key_h[50];
	int key_ind_h[600];
	int *key_ind_d;
	int currind;
	int *curr_ind_h = &currind;
	int *curr_ind_d;



	/*******************************Data and Data Size*********************************************************/
	// When searching Bilbo in OriginalText, set key_ind_h[600].
	// When searching 'Bilbo' in EditedText, set key_ind_h[12000].
	pFile = fopen("OriginalText.txt", "rb");
	if (pFile == NULL)
	{
		printf("Cannot open txt file!\n");
		exit(1);
	}
	fseek(pFile, 0, SEEK_END);
	lSize = ftell(pFile);
	rewind(pFile);

	data_h = new char[lSize];	// allocate memory on host
	cudaMalloc((void **)&data_d, lSize);	// allocate memory on device	- this is the first CUDA call so it takes huge amount of time

	StartCounter();

	numB = fread(data_h, 1, lSize, pFile);
	cudaMemcpy(data_d, data_h, lSize, cudaMemcpyHostToDevice);	// copy data to device memory
	/************************************************************************************************************/

	/******************************key and key_size*************************************************************/
	//printf("What do you want to search: ");
	//scanf("%[^\n]s ", key_h);	// enter the key
	strcpy(key_h, "Bilbo");
	key_size = strlen(key_h);

	//StartCounter();

	cudaMalloc((void **)&key_d, key_size);
	cudaMemcpy(key_d, key_h, key_size, cudaMemcpyHostToDevice);
	/**********************************************************************************************************/

	/****************************key indices and current indice*************************************************/
	key_ind_size = sizeof(key_ind_h);
	memset(key_ind_h, 0, key_ind_size);
	cudaMalloc((void **)&key_ind_d, key_ind_size);
	cudaMemcpy(key_ind_d, key_ind_h, key_ind_size, cudaMemcpyHostToDevice);

	*curr_ind_h = 0;
	cudaMalloc((void **)&curr_ind_d, 4);
	cudaMemcpy(curr_ind_d, curr_ind_h, 4, cudaMemcpyHostToDevice);
	/***********************************************************************************************************/

	int block_size = 1024;
	int n_blocks = lSize / block_size + (lSize%block_size == 0 ? 0 : 1);

	parallelGrep << < n_blocks, block_size>> > (data_d, lSize, key_d, key_size, key_ind_d, curr_ind_d);
	
	// For debug
	/*cudaError_t err;
	err = cudaPeekAtLastError();*/

	cudaMemcpy(key_ind_h, key_ind_d, key_ind_size, cudaMemcpyDeviceToHost);


	printf("\n Elapsed Time: ");
	std::cout << GetCounter() << " ms" << std::endl;


	/*****************	To print lines with key string	************************/
	//int firstChar;
	//int pIter = 0;
	//while (key_ind_h[pIter] != 0)
	//{
	//	firstChar = key_ind_h[pIter];
	//	// Find the first character of the line
	//	while (data_h[firstChar - 1] != '\n')
	//	{
	//		firstChar--;
	//	}
	//	// print till the last character of the line
	//	while (data_h[firstChar] != '\n')
	//	{
	//		printf("%c", data_h[firstChar]);
	//		firstChar++;
	//	}
	//	printf("\n");
	//	pIter++;
	//}
	//printf("\n Elapsed Time: ");
	//std::cout << GetCounter() <<" ms"<< std::endl;
	/*****************************************************************************/

	delete[] data_h;
	cudaFree(data_d);
	cudaFree(key_d);
	cudaFree(key_ind_d);
	cudaFree(curr_ind_d);
	return 0;
}
