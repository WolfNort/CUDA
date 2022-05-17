
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cuda.h"
#include "curand_kernel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <conio.h>
#include <math.h>
#include "cublas_v2.h"
static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

const int BLOCK = 20;

void print_mtx(int* matr, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matr[i + n * j]);
        }
        printf("\n");
    }
}

void print_mtx_double(double* matr, int n)
{
    printf("\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2lf ", matr[i + n * j]);
        }
        printf("\n");
    }
}

__device__ int generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    int RANDOM = curand(&localState);
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void get_matr(int* matr, int* sub_matr, int count, int ind_row, int ind_col, size_t pitch)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < count) && (row < count))
    {
        if (row != ind_row)
        {
            if (col != ind_col)
            {
                if ((row > ind_row) && (col > ind_col))
                    *(((int*)(((char*)sub_matr) + ((row - 1) * pitch))) + col - 1) = *(((int*)(((char*)matr) + (row * pitch))) + col);
                else if (row > ind_row)
                    *(((int*)(((char*)sub_matr) + ((row - 1) * pitch))) + col) = *(((int*)(((char*)matr) + (row * pitch))) + col);
                else if (col > ind_col)
                    *(((int*)(((char*)sub_matr) + (row * pitch))) + col - 1) = *(((int*)(((char*)matr) + (row * pitch))) + col);
                else
                    *(((int*)(((char*)sub_matr) + (row * pitch))) + col) = *(((int*)(((char*)matr) + (row * pitch))) + col);
            }

        }
    }
}

__host__ int result_determinate(int* matr_cuda, int* number, size_t pitch)
{
    int count = *number;
    dim3 dimBlock(count, count);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, (count + dimBlock.y - 1) / dimBlock.y);
    int* temp_matr_CUDA = NULL;
    int* matr = new int[count*count];
    int temp = 0;   //временная переменная для хранения определителя
    int k = 1;      //степень

    HANDLE_ERROR(cudaMemcpy2D(matr, count * sizeof(int), matr_cuda, pitch, count * sizeof(int), count, cudaMemcpyDeviceToHost));

    if (count < 1)
    {
        printf("Size isn't True");
        return 0;
    }
    else if (count == 1)
        temp = matr[0 + count * 0];
    else if (count == 2)
    {
        temp = matr[0 + count * 0] * matr[1 + count * 1] - matr[1 + count * 0] * matr[0 + count * 1];
       // HANDLE_ERROR(cudaFree(matr));
    }
    else
    {
        for (int i = 0; i < count; i++)
        {
            int m = count - 1;
            HANDLE_ERROR(cudaMallocPitch(&temp_matr_CUDA, &pitch, m * sizeof(int), m));
            get_matr << <dimGrid, dimBlock >> > (matr_cuda, temp_matr_CUDA, count, 0, i, pitch);

            int value = matr[0 * count + i];
            temp = temp + k * value * result_determinate(temp_matr_CUDA, &m, pitch);
            k = -k;
            HANDLE_ERROR(cudaFree(temp_matr_CUDA));
            
        } 
    }
    
    return temp;
}
__global__ void push_obr_matrix(double* obr_matr, int determinate, int count, size_t pitch)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < count) && (row < count))
    {
        double start = *(((double*)(((char*)obr_matr) + ((row)*pitch))) + col);
        double value = start / double(determinate);
        *((double*)((char*)obr_matr + (row * pitch)) + col) = value;
    }
}

__global__ void setup_kernel(curandState* state, unsigned long seed, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col < n) && (row < n))
    {
        int index = col * n + row;
        curand_init(seed, index, 0, &state[index]);
    }
}

__global__ void set_mtx(curandState* globalState, int* matr, int n, size_t pitch)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index = col * n + row;
    if ((col < n) && (row < n))
    {
        int number = generate(globalState, index) % 10;
        if (number == 0) number++;
        *(((int*)(((char*)matr) + (row * pitch))) + col) = abs(number);
    }
}

__global__ void transpose(double* matr, double* transpose_matr, int count, size_t pitch)
{
    __shared__ double temp[BLOCK][BLOCK];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col < count) && (row < count))
        temp[threadIdx.y][threadIdx.x] = *(((double*)(((char*)matr) + (row * pitch))) + col);

    __syncthreads();
    col = blockIdx.x * blockDim.x + threadIdx.x;
    row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((col < count) && (row < count))
        *((double*)((char*)transpose_matr + (col * pitch)) + row) = temp[threadIdx.y][threadIdx.x];
}

__host__ int main() //host, так как ядро вызывается из host
{
    curandState* devStates;
    cudaEvent_t start, stop;
    float gpuTime = 0.0;
    
    double step = 1.0;
    int count, det = 0;
    scanf("%d", &count);
    HANDLE_ERROR(cudaMalloc(&devStates, count * count * sizeof(curandState)));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* matr = new int[count * count];
    double* obr_matr = new double[count * count];
    double* t_obr_matr = new double[count * count];
    double* t_obr_matr_cuda = NULL;
    int* matr_cuda = NULL;
    double* obr_matr_cuda = NULL;
    int* temp_matr = NULL;

    size_t pitch;
    dim3 dimBlock(count, count);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x, (count + dimBlock.y - 1) / dimBlock.y);

    HANDLE_ERROR(cudaMallocPitch(&obr_matr_cuda, &pitch, count * sizeof(double), count));
    HANDLE_ERROR(cudaMallocPitch(&matr_cuda, &pitch, count * sizeof(int), count));
    //cudaMallocPitch(&temp_matr, &pitch, count * sizeof(int), count);
    HANDLE_ERROR(cudaMallocPitch(&t_obr_matr_cuda, &pitch, count * sizeof(double), count));
    
    cudaEventRecord(start, 0);
    setup_kernel << <dimGrid, dimBlock >> > (devStates, unsigned(time(NULL)), count);
    set_mtx << <dimGrid, dimBlock >> > (devStates, matr_cuda, count, pitch);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);
    printf("\ntime on GPU = %.5f miliseconds\n", gpuTime);
    HANDLE_ERROR(cudaMemcpy2D(matr, count * sizeof(int), matr_cuda, pitch, count * sizeof(int), count, cudaMemcpyDeviceToHost));
    print_mtx(matr, count);
    int m = count - 1;
    HANDLE_ERROR(cudaMallocPitch(&temp_matr, &pitch, m * sizeof(int), m));
    for (int i = 0; i < count; i++)
    {
        for (int j = 0; j < count; j++)
        {
            get_matr << <dimGrid, dimBlock >> > (matr_cuda, temp_matr, count, i, j, pitch);
            if ((i + j + 2) % 2) step = -1;
            else step = 1;
            obr_matr[i * count + j] = step * result_determinate(temp_matr, &m, pitch);
            if (i == 0 && j < count)
                det += matr[i * count + j] * obr_matr[i * count + j];
        }
    }
    HANDLE_ERROR(cudaFree(temp_matr));
    printf("determinate = %d", det);

    cudaMemcpy2D(obr_matr_cuda, pitch, obr_matr, count * sizeof(double), count * sizeof(double), count, cudaMemcpyHostToDevice);
    push_obr_matrix << <dimGrid, dimBlock >> > (obr_matr_cuda, det, count, pitch);
    transpose << <dimGrid, dimBlock >> > (obr_matr_cuda, t_obr_matr_cuda, count, pitch);
    cudaMemcpy2D(t_obr_matr, count * sizeof(double), t_obr_matr_cuda, pitch, count * sizeof(double), count, cudaMemcpyDeviceToHost);
   
    print_mtx_double(t_obr_matr, count);

    cudaFree(devStates);
    cudaFree(matr);
    cudaFree(matr_cuda);
    cudaFree(obr_matr_cuda);
    cudaFree(t_obr_matr);
    cudaFree(temp_matr);
    cudaFree(obr_matr);
    cudaFree(t_obr_matr_cuda);
    return 0;
}
