#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;


__global__ void BitonicMerge(int size, unsigned int* arrayDevice,int numStages)
{
 
    unsigned const int GlobalThreadId = blockIdx.x*blockDim.x + threadIdx.x; 
    unsigned const int BlockThreadId  = threadIdx.x;
    //const int mem = blockIdx.x * 2;
    __shared__ int sharedDeviceArray[1024];

    // Each Block will write to its Shared Memory
    sharedDeviceArray[BlockThreadId] = arrayDevice[GlobalThreadId];


    for( int stage =  1 ;  stage < numStages ; stage++)
    {
        int substage = stage;
       // if(ThreadId == 0 ) printf("@@@@@@@-------STAGE %d -----@@- \n",stage);
        while(substage)
        {
        //    if(ThreadId == 0 ) printf(" -------------STAGE %d --- SUBSTAGE %d-------------------- \n",stage,substage);
            int Displace =  1 << substage - 1 ;         // NUmber of elements to Be Displaced 
             if (BlockThreadId < size)
            {
                // printf("   ccycle : %d  Thread id : %d \n",ThreadCycle,ThreadId);
                //int N = stage;
                if(!(BlockThreadId & (1 << (substage -1))))      // Nth Bit is checked for Entry Criteria
                {
                    if(  ( BlockThreadId & (1 << (stage)) )   )   // N+1th bit Set as 1 , So Decending
                    {
                        // printf("ThreadId : %d ,cycle %d  ,arr[0] - %d ,arr[1] - %d\n",ThreadId_new,ThreadCycle,sharedDeviceArray_1[ThreadId], sharedDeviceArray_2[ThreadId]);
                        int arry_1 = sharedDeviceArray[BlockThreadId] ; int arry_2 = sharedDeviceArray[BlockThreadId + Displace];
                        if(arry_1 < arry_2)
                        {
                            sharedDeviceArray[BlockThreadId]               = arry_2;
                            sharedDeviceArray[BlockThreadId + Displace]    = arry_1;
                        }
                    }
                    else                          // N+1th bit Set as 0  , So Asscending
                    {
                        // printf(".ThreadId : %d ,cycle %d  ,arr[0] - %d ,arr[1] - %d\n",ThreadId_new,ThreadCycle,sharedDeviceArray_1[ThreadId], sharedDeviceArray_2[ThreadId]);
                        int arry_1 = sharedDeviceArray[BlockThreadId] ; int arry_2 = sharedDeviceArray[BlockThreadId + Displace];
                        if(arry_1 >  arry_2)
                        {
                            sharedDeviceArray[BlockThreadId]            = arry_2;
                            sharedDeviceArray[BlockThreadId + Displace] = arry_1;
                            // cout << "       K,K+disp - (" << k <<","<<k+Displace<<") " << " Accent" <<endl;
                        }
                    }
                }
            // cout << endl;
            }
            //    __syncthreads();
            substage--;
            __syncthreads();
        }
        // __syncthreads();
    }

    __syncthreads();

    for ( int stage = numStages ;  stage > 0 ; stage--)
    {
        //Pick NUmbers which has 'stage'th bit set as 0
        int Displace = 1 << (stage-1)  ;
        if (BlockThreadId < size)
        {
            if( !(BlockThreadId  & ( 1 << stage - 1 ) ) )
            {
                int arry_1 = sharedDeviceArray[BlockThreadId] ; int arry_2 = sharedDeviceArray[BlockThreadId + Displace];
                if(arry_1  > arry_2)
                {
                    sharedDeviceArray[BlockThreadId] = arry_2 ;
                    sharedDeviceArray[BlockThreadId + Displace] = arry_1;
                }
            }
        }
            // __syncthreads();
        __syncthreads();
    }

    __syncthreads();


    // Copy the Device Array to Host 
    arrayDevice[GlobalThreadId] = sharedDeviceArray[BlockThreadId];

}

__global__ void MergeSort(int numThreadsPerBlock, unsigned int* arrayDevice, int numBlocks , int stage)
{
     int chunkSize =  numThreadsPerBlock * (1<<stage);
    const int totalArraySize = chunkSize * 2;
    unsigned const int displacement = chunkSize;  //  2^(stage+1);

    unsigned const int GlobalThreadId = blockIdx.x*blockDim.x + threadIdx.x; 
    unsigned const int BlockThreadId  = threadIdx.x;
    unsigned const int N_blockCycle   = (1 << stage);
    
    
    
    if(GlobalThreadId == 0)   printf("N_BlockCycle : %d , displacement :  %d, totalarraysize : %d, chunk : %d" , N_blockCycle,
            displacement,totalArraySize,chunkSize );
    unsigned int iteration = 0;
    while ( iteration < chunkSize )
    {
        for ( int blockcycle = 0 ; blockcycle < N_blockCycle ; blockcycle ++)
        {
            unsigned int leftIndex = BlockThreadId   + numThreadsPerBlock*blockcycle + totalArraySize*blockIdx.x;
            unsigned int rightIndex = BlockThreadId  + numThreadsPerBlock*blockcycle + totalArraySize*blockIdx.x  + displacement;
            unsigned int blockcycleThreadId  = BlockThreadId + blockcycle*numThreadsPerBlock;
            
            // printf("BlockCycle :  %d , BlockThread: %d ,BlockID: %d, totalarray : %d  globalTid : %d , leftI : %d, right : %d \n",blockcycle,
            //         BlockThreadId,blockIdx.x,totalArraySize,GlobalThreadId , leftIndex ,rightIndex);
            if(iteration)     // LASt BIT is 1 - ODD Iteration - Shift Iteration (between threads)
            {
                // __syncthreads();
                if( blockcycleThreadId + iteration < chunkSize  )
                {
                
                   int  left_array = arrayDevice[leftIndex + iteration];
                   int  right_array = arrayDevice[rightIndex];

                //    
                    if(left_array > right_array)
                    {
                        arrayDevice[leftIndex + iteration] = right_array;
                        arrayDevice[rightIndex]    = left_array;
                    }
                }

                // __syncthreads();
            }
            
            else              // Last bit is even - even Iteration - Swap iteration ( inside thread) 
            {
               int left_array = arrayDevice[leftIndex];
               int right_array = arrayDevice[rightIndex];
               
                if(left_array > right_array)
                {
                    // printf("even iteration numT: %d blockid : %d blockcyc : %d  globTid %d  arD[%d] = %d , arD[%d] = %d \n",numThreadsPerBlock,BlockThreadId,blockcycle,GlobalThreadId,leftIndex,left_array,rightIndex,right_array);
                    arrayDevice[leftIndex] = right_array;
                    arrayDevice[rightIndex] = left_array;
                    
                }
               
            }
            __syncthreads();
  
        }

        __syncthreads();
        iteration ++;
    }

 





}



// #define N 16
#define NUM_THREADS 1024

int main(int argc, char** argv)
{   

    const long int size =   stol(argv[1]);
    
    // int array[16] = {10,20,5,9,3,8,12,14,90,0,60,40,23,35,95,18};

    unsigned int* arrayDevice;
    unsigned int* base_array = new unsigned int[size];  // For Comparision

    const int sizeofA = sizeof(int)*size;
   
    // Polulate random values in the array 
    
    unsigned int* array = new unsigned int[size];
    srand(time(0));
    for ( int i =  0 ; i < size ; i++)
    {
        array[i] = rand() % 100000;
        base_array[i] = array[i] ;
    }
    cout <<endl;

    struct timespec start,end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);

    // Allocate Memory in Device 
    cudaMalloc((void **) &arrayDevice, sizeofA);

    // Copy the Array from Host to Device
    cudaMemcpy(arrayDevice,&array[0],sizeofA,cudaMemcpyHostToDevice);


    // ********* CALL FOR BITONIC SORT FOR INDIVIDIAL ELEMENTS ********************* //
    const int numThreadsPerBlock = NUM_THREADS;
    const int numBlocks = size/NUM_THREADS;
    int chunkSize = NUM_THREADS;
    
    dim3 dimGrid(numBlocks);     //Set the Dimensions of the Block
    dim3 dimBlock(numThreadsPerBlock);

    const int numStages = log2(chunkSize);  
   
    BitonicMerge<<<dimGrid,dimBlock>>>(chunkSize,arrayDevice,numStages);

    cudaDeviceSynchronize();

    // ***** END OF BITONIC SORT CALL ****************************************** //
    
    // ****** CALL FOR MERGE SORT **********************************************//
    int N_stages = log2(numBlocks);
    cout << " NSTAGES : " <<N_stages <<endl;

    int numBlocks_sort = numBlocks;
    for ( int stage = 0 ;stage < N_stages ; stage++)
    {
         numBlocks_sort = numBlocks_sort/2;
        cout << " NUm Blocks : "<< numBlocks_sort <<endl;
        dim3 dimGrid1(numBlocks_sort);     //Set the Dimensions of the Block
        dim3 dimBlock1(numThreadsPerBlock);

        MergeSort<<<dimGrid1,dimBlock1>>>(numThreadsPerBlock,arrayDevice,numBlocks_sort,stage);
        cudaDeviceSynchronize();
    }




    cudaMemcpy(&array[0],arrayDevice,sizeofA,cudaMemcpyDeviceToHost);

    cudaFree(arrayDevice);
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
    

    std::sort(base_array,base_array+size);

    bool compareFlag = true;

    for ( int i = 0 ; i < size ; i++)
        if( array[i] != base_array[i]){
            compareFlag = false;
            break;
        }

    // for (int i = 0 ; i < size ; i++)
    //     cout << array[i] << "   "  ;
    // cout <<endl;
    
    if(compareFlag){
        // cout << " Time for Iteration : " << (end_iter.tv_sec - start_iter.tv_sec) + (end_iter.tv_nsec -  start_iter.tv_nsec)*1e-9 <<endl;
        cout << " Time for Execution : " << (end.tv_sec - start.tv_sec) + (end.tv_nsec -  start.tv_nsec)*1e-9 <<endl;
    }


    return 0 ;

}

