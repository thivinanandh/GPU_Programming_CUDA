#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;


__global__ void BitonicMerge(int size, unsigned int* arrayDevice,int numStages,int N_ThreadCycle)
{
 
    unsigned const long int ThreadId = blockIdx.x*blockDim.x + threadIdx.x; 
    //const int mem = blockIdx.x * 2;
    for( int stage =  1 ;  stage < numStages ; stage++)
    {
        int substage = stage;
       // if(ThreadId == 0 ) printf("@@@@@@@-------STAGE %d -----@@- \n",stage);
        while(substage)
        {
        //    if(ThreadId == 0 ) printf(" -------------STAGE %d --- SUBSTAGE %d-------------------- \n",stage,substage);
            int Displace =  1 << substage - 1 ;         // NUmber of elements to Be Displaced 
            // cout << " displace : " << Displace <<endl; 
            if(N_ThreadCycle == 0) N_ThreadCycle++;
            for (int ThreadCycle = 0 ; ThreadCycle < N_ThreadCycle  ; ThreadCycle++)
            {
                //COPY  THE ARRAYS  to Shared memory
                
                unsigned long int ThreadId_new = ThreadId + ThreadCycle*blockDim.x;             

                if (ThreadId_new < size)
                {
                    // printf("   ccycle : %d  Thread id : %d \n",ThreadCycle,ThreadId);
                    //int N = stage;
                    if(!(ThreadId_new & (1 << (substage -1))))      // Nth Bit is checked for Entry Criteria
                    {
                        if(  ( ThreadId_new & (1 << (stage)) )   )   // N+1th bit Set as 1 , So Decending
                        {
                            // printf("ThreadId : %d ,cycle %d  ,arr[0] - %d ,arr[1] - %d\n",ThreadId_new,ThreadCycle,sharedDeviceArray_1[ThreadId], sharedDeviceArray_2[ThreadId]);
                            int arry_1 = arrayDevice[ThreadId_new] ; int arry_2 = arrayDevice[ThreadId_new + Displace];
                            if(arry_1 < arry_2)
                            {
                                arrayDevice[ThreadId_new]               = arry_2;
                                arrayDevice[ThreadId_new + Displace]    = arry_1;
                            }
                        }
                        else                          // N+1th bit Set as 0  , So Asscending
                        {
                            // printf(".ThreadId : %d ,cycle %d  ,arr[0] - %d ,arr[1] - %d\n",ThreadId_new,ThreadCycle,sharedDeviceArray_1[ThreadId], sharedDeviceArray_2[ThreadId]);
                            int arry_1 = arrayDevice[ThreadId_new] ; int arry_2 = arrayDevice[ThreadId_new + Displace];
                            if(arry_1 >  arry_2)
                            {
                                arrayDevice[ThreadId_new]            = arry_2;
                                arrayDevice[ThreadId_new + Displace] = arry_1;
                               // cout << "       K,K+disp - (" << k <<","<<k+Displace<<") " << " Accent" <<endl;
                            }
                        }
                    }
                // cout << endl;
                }
            //    __syncthreads();
                        // printf("t - %d ,sh1 : %d , sh2 : %d \n",ThreadId,sharedDeviceArray_1[ThreadId],sharedDeviceArray_2[ThreadId]);
            }
            substage--;
            // __syncthreads();
        }
        __syncthreads();
    }

}

__global__ void BitonicSort(int size, unsigned int* arrayDevice,int numStages,int N_ThreadCycle)
{
    unsigned const int ThreadId = blockIdx.x*blockDim.x + threadIdx.x;
  
    for ( int stage = numStages ;  stage > 0 ; stage--)
    {
        //Pick NUmbers which has 'stage'th bit set as 0
        int Displace = 1 << (stage-1)  ;
        if(N_ThreadCycle == 0) N_ThreadCycle++; 
        for (int ThreadCycle = 0 ; ThreadCycle < N_ThreadCycle  ; ThreadCycle++)
        {
            unsigned long int ThreadId_new = ThreadId + ThreadCycle*blockDim.x;

            if (ThreadId_new < size)
            {
                if( !(ThreadId_new  & ( 1 << stage - 1 ) ) )
                {
                    int arry_1 = arrayDevice[ThreadId_new] ; int arry_2 = arrayDevice[ThreadId_new + Displace];
                    if(arry_1  > arry_2)
                    {
                        arrayDevice[ThreadId_new] = arry_2 ;
                        arrayDevice[ThreadId_new + Displace] = arry_1;
                    }
                }
            }
            // __syncthreads();
        }
        __syncthreads();
    }

}


// #define N 16
#define NUM_THREADS 1024

int main(int argc, char** argv)
{   

    const long int size = stol(argv[1]);
    
    // int array[16] = {10,20,5,9,3,8,12,14,90,0,60,40,23,35,95,18};

    unsigned int* arrayDevice;
    unsigned int* base_array = new unsigned int[size];  // For Comparision

    const int sizeofA = sizeof(int)*size;
   
    // Polulate random values in the array 
    
    unsigned int* array = new unsigned int[size];
    srand(time(0));
    for ( int i =  0 ; i < size ; i++)
    {
        array[i] = rand() % 10000;
        base_array[i] = array[i] ;
    }
    cout <<endl;

    struct timespec start,end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);

    // Allocate Memory in Device 
    cudaMalloc((void **) &arrayDevice, sizeofA);

    // Copy the Array from Host to Device
    cudaMemcpy(arrayDevice,&array[0],sizeofA,cudaMemcpyHostToDevice);

    int numThreadsPerBlock = NUM_THREADS;
    int numBlocks = 1;


    //Set the Dimensions of the Block
    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);


    // Thread Cycle is number of times each thread block needs to perform an Operation for a given set of array
    int N_ThreadCycle = size/numThreadsPerBlock ;
    // cout << " N-THREAD CYCLE : " << N_ThreadCycle <<endl;

    struct timespec start_iter,end_iter;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start_iter);

    const int numStages = log2(size);
   
    BitonicMerge<<<dimGrid,dimBlock>>>(size,arrayDevice,numStages,N_ThreadCycle);

    BitonicSort<<<dimGrid,dimBlock>>>(size,arrayDevice,numStages,N_ThreadCycle);

    cudaDeviceSynchronize();

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end_iter);
    cudaMemcpy(&array[0],arrayDevice,sizeofA,cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);

    cudaFree(arrayDevice);


    
  
    

    std::sort(base_array,base_array+size);

    bool compareFlag = true;

    for ( int i = 0 ; i < size ; i++)
        if( array[i] != base_array[i]){
            compareFlag = false;
            break;
        }

    // for (int i = 0 ; i < N ; i++)
    //     cout << array[i] << "   "  << base_array[i] <<endl;
    // cout <<endl;
    
    if(compareFlag){
        cout <<endl;
        cout << " Time for Iteration : " << (end_iter.tv_sec - start_iter.tv_sec) + (end_iter.tv_nsec -  start_iter.tv_nsec)*1e-9 <<endl;
        cout << " Time for Execution : " << (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9 <<endl;
    }
    cout << endl;


    return 0 ;

}

