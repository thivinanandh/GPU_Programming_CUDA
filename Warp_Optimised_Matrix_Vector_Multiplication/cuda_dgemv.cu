#include<iostream>
#include<cmath>
#include <ctime>
#include<fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;

__global__ void GPU_dgemv(double* d_A,int d_row,int d_col,double* d_x, double* d_b3)
{
    const unsigned int ThreadId = blockIdx.x*blockDim.x + threadIdx.x;

    if(ThreadId < d_row)
    {
        double sum = 0;
        for ( unsigned int k = 0 ; k < d_col ; k++){
            unsigned int index = k + d_col*ThreadId;
            sum += d_A[index]*d_x[k];
        }
        d_b3[ThreadId] = sum;
    }
}


__global__ void GPU_dgemv_3(double* d_A,int d_row,int d_col,double* d_x, double* d_b1 )
{
    __shared__ double ans3[1056];
    unsigned long int WARP            = 32;
    unsigned long int ThreadId        = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int localWarpId          = ThreadId % WARP;
    unsigned long int localThreadId   = threadIdx.x;
    unsigned int warpId               = ThreadId / WARP ;
    int unsigned long threadRow       = d_col/WARP;
    
    
    // if(!ThreadId)  printf("Thread Ro :  %d \n ",threadRow);
    int counter = 0;
    ans3[threadIdx.x] = 0;
    for ( int k = 0 ; k < threadRow ; k++){
        long int col                = (threadIdx.x % WARP) + (k*WARP);
        unsigned long int index     = col + warpId*d_col;
        ans3[threadIdx.x]         += d_A[index]*d_x[col];
    }

    for ( int stage = 0 ; stage < 5 ; stage++)
    {
        int elemShift           =  1 << (stage);
        ans3[threadIdx.x]       += ans3[threadIdx.x + elemShift];
        __syncthreads();
    }

    
    if(threadIdx.x % WARP == 0)
        d_b1[warpId] = ans3[threadIdx.x];

//     cudaFree(ans3);
}




__global__ void GPU_dgemv_2(double* d_A,int d_row,int d_col,double* d_x, double* d_b2)
{
    __shared__ double ans[1024];

    const unsigned long int WARP     = 32;
    const unsigned long int ThreadId = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned long int localThreadId = threadIdx.x;
    const unsigned long int warpId   = ThreadId / WARP ;
    int unsigned long threadRow      = d_col/WARP;
    
    // if(!ThreadId)  printf("Thread Ro :  %d \n ",threadRow);
    int counter = 0;
    ans[localThreadId] = 0;
    for ( int k = 0 ; k < threadRow ; k++){
        long int col                = (threadIdx.x % WARP) + (k*WARP);
        unsigned long int index     = col + warpId*d_col;
        ans[localThreadId]         += d_A[index]*d_x[col];
        // printf(" Thread : %d , Thredid.x : %d,  col : %d , threadrow : %d, warpid: %d \n" ,ThreadId,threadIdx.x,col,k,warpId );
    }
    

    if(threadIdx.x % WARP == 0)
    {
        double sum = 0;
        long int start  = localThreadId;
        long int end    = start + WARP;
        
        for ( int i =  start ; i < end ; i++){
            sum += ans[i];
        }
        d_b2[warpId] = sum;
    }
}


int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cout << " NOt enough Input Parameters in the input " << endl;
        cout << " Param 1 -  num rows, Param 2 - num cols" <<endl;
        cout << " Param 3 -  numThreadsPerBlock , Param 4 - numBlocks" <<endl;
        exit(0);
    }
    // Declare a 2d Array as an 1d Flattened Array
    const int h_row                 =  stoi(argv[1]);
    const int h_col                 =  stoi(argv[2]);

    int numThreadsPerBlock, numBlocks;
    int size        = h_row * h_col;
    cout << " ROW :   " <<h_row <<endl;
    double* A       = new double[size]();
    double* x       = new double[h_col]();
    double* b_1     = new double[h_row]();
    double* b_3     = new double[h_row]();
    double* b_2     = new double[h_row]();
    double* b_verf     = new double[h_row]();
    
    
    double *d_A, *d_x,*d_b1,*d_b2,*d_b3;
    cudaMalloc((void **) &d_A,size*sizeof(double));
    cudaMalloc((void **) &d_x,h_col*sizeof(double));
    cudaMalloc((void **) &d_b1,h_row*sizeof(double));
    
    for ( int i = 0 ; i < size ; i++) A[i] = rand() % 30;
    for ( int i = 0 ; i < h_col ; i++) x[i] = rand() % 30 ;



    // ------------- TYPE 1 - Optimisation Code ---------------- //
    if (h_row*32 > 1024){
        numBlocks           = ceil (double(h_row*32/1024.0)) ;
        numThreadsPerBlock  = 1024;
    }
    else
    {
        numBlocks           = 1;
        numThreadsPerBlock  = h_row*32;
    }

    cudaMemcpy(d_A,A,size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x,h_col*sizeof(double),cudaMemcpyHostToDevice);

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);
    // cout << " PROB -2 : NumThreadsperblock : " <<numThreadsPerBlock << " NUm Blocks : " << numBlocks <<endl;
    struct timespec start,end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
    
    GPU_dgemv_3<<<dimGrid,numThreadsPerBlock>>>(d_A,h_row,h_col,d_x,d_b1);
    
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);

        
    cudaMemcpy(b_1,d_b1,h_row*sizeof(double),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_b1);
    

    double GPU_time1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9;

    // ------------- TYPE 1 - Optimisation Code END ---------------- //
    
    //--------------- TYPE 2 - Optimisation Code -------------------- //
    cudaMalloc((void **) &d_b2,h_row*sizeof(double));
    
    if (h_row*32 > 1024){
        numBlocks           = ceil (double(h_row*32/1024.0)) ;
        numThreadsPerBlock  = 1024;
    }
    else
    {
        numBlocks           = 1;
        numThreadsPerBlock  = h_row*32;
    }

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
    GPU_dgemv_2<<<numBlocks,numThreadsPerBlock>>>(d_A,h_row,h_col,d_x,d_b2);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);

    cudaMemcpy(b_2,d_b2,h_row*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    double GPU_time2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9;
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
    cudaFree(d_b2);
    //--------------- TYPE 2 - Optimisation Code END -------------------- //
   
    // --------  TYPE 3 - Optimsation CODE ------------------------------- //

    cudaMalloc((void **) &d_b3,h_row*sizeof(double));
    
    // Override the Number of Thread block Value
    if (h_row > 1024){
        numBlocks           = ceil (double(h_row/1024.0)) ;
        numThreadsPerBlock  = 1024;
    }
    else
    {
        numBlocks           = 1;
        numThreadsPerBlock  = h_row;
    }
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
    GPU_dgemv<<<dimGrid,dimBlock>>>(d_A,h_row,h_col,d_x,d_b3);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);

    // Copy the answer to host
    cudaMemcpy(b_3,d_b3,h_row*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_b3);
    
    double GPU_time3 = (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9;
    
    // --------------------- OPTIMISATION LEVEL 3 - CODE - END --------------------- //
    
    // ----- Clearing CUda Variables -------- //
    cudaFree(d_A);
    cudaFree(d_x);
    cudaDeviceReset();
    
    // ---------------------- CHECKING CODE --------------------------------------- //
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);
    for ( int row = 0 ; row < h_row ; row++)
    {
        double sum = 0;
        unsigned int shift  = h_col*row;
        for ( int col =0 ; col < h_col ; col++)
            sum += A[col + shift] * x[col];
        b_verf[row] = sum;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
    double CPU_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9;
    
    
    int verifyFlag = 0;
    for ( int k = 0 ; k < h_row ; k++)
        if( b_verf[k] != b_1[k] || b_verf[k] != b_3[k] || b_verf[k] != b_2[k] )
        {
            verifyFlag = 1;
            if(b_verf[k] != b_1[k] ) cout << " K Val : "<<k <<endl;
            break;
        }
   
   // ---------------------- CHECKING CODE --------------------------------------- //
    // for ( int i = 0 ; i < h_row; i++)
    //     cout << b_verf[i] <<" \t";
    // cout<<endl;


    // for ( int i = 0 ; i < h_row; i++)
    //     cout << b_1[i] <<" \t";
    // cout<<endl;

    // for ( int i = 0 ; i < h_row; i++)
    //     cout << b_3[i] <<" \t";
    // cout<<endl;
    std::cout << std::setprecision(8);
    if(!verifyFlag)
    {
        cout << ""<< GPU_time1 <<"\t";
        cout << ""<< GPU_time2 <<"\t";
        cout << ""<< GPU_time3 <<"\t";
        cout << ""<< CPU_time <<"\t";
        cout << ""<< CPU_time / GPU_time1 <<"\t";
        cout << ""<< GPU_time3 / GPU_time1 <<"\t";
        cout << ""<< GPU_time3 / GPU_time2 <<endl;
    }

    return 0;
}



