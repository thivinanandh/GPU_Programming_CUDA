#include<iostream>
#include<cmath>
#include <ctime>
#include<fstream>

using namespace std;




__global__ void Prefix_Sum(int N_, int N_Levels,int N_ThreadCycle, int* arrayDevice)
{
   
    
   unsigned int ThreadId = blockIdx.x*blockDim.x + threadIdx.x;
   
   // Forward Sweep
    for (int level = 0 ; level <  N_Levels; level++)
    {
        unsigned int elemJump = 1; 
        elemJump = elemJump << (level+1); // pow(2,level+1);

        for (int ThreadCycle = 0 ; ThreadCycle < N_ThreadCycle ; ThreadCycle++)
        {
            if( ((ThreadId+1) + ThreadCycle*blockDim.x) % elemJump == 0 )
            {
                unsigned int elem =   ((ThreadId+ThreadCycle*blockDim.x) ^ (1 << (level)));
                arrayDevice[ThreadId + ThreadCycle*blockDim.x] += arrayDevice[elem];
            }
        }
        __syncthreads();
    }
    
   // backward Sweep
    if(ThreadId ==  0)  arrayDevice[N_ - 1] = 0;

    for (int level =  N_Levels -1 ; level >=0 ; level--)
    {
        unsigned int elemJump = 1; 
        elemJump = elemJump << (level+1); // pow(2,level+1);
        for (int ThreadCycle = 0 ; ThreadCycle < N_ThreadCycle ; ThreadCycle++)
        {
            if( ((ThreadId+1) + ThreadCycle*blockDim.x) % elemJump == 0   )
            {
                unsigned int elem =   ((ThreadId + ThreadCycle*blockDim.x) ^ (1 << (level)));    // Bit Inversion 
                int temp = arrayDevice[ThreadId + ThreadCycle*blockDim.x];
                arrayDevice[ThreadId + ThreadCycle*blockDim.x] += arrayDevice[elem];
                arrayDevice[elem] = temp;
            }
        }
        __syncthreads();

    }


}


#define N 16384
#define NUM_THREADS 1024


int main()
{
    int* array = new int[N];

    int* arrayDevice;

    int N_Levels = log2(N);
    cout << " N Lev : " << N_Levels<<endl;

    for (int i = 0 ; i < N ; i++ ) array[i] = i;

    int sizeofA = sizeof(int)*N;
    
    // Allocate Memory on the GPU(DEVICE) 
    cudaMalloc((void **) &arrayDevice, sizeofA);

    // Copy the Array from Host to Device
    cudaMemcpy(arrayDevice,array,sizeofA,cudaMemcpyHostToDevice);

    int numThreadsPerBlock = NUM_THREADS;
    int numBlocks = 1;

    dim3 dimGrid(numBlocks);
    dim3 dimBlock(numThreadsPerBlock);

    int N_ThreadCycle = N/numThreadsPerBlock;
    cout << " N-THREAD CYCLE : " << N_ThreadCycle <<endl;

    //TIME THE PROCESS
    struct timespec start,end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&start);

    Prefix_Sum<<<dimGrid,dimBlock>>>(N,N_Levels,N_ThreadCycle,arrayDevice);
    
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&end);
    cout<<(end.tv_sec-start.tv_sec) +((end.tv_nsec-start.tv_nsec)*1e-9)<< " in sec "<<endl;

    cudaMemcpy(array,arrayDevice,sizeofA,cudaMemcpyDeviceToHost);

    if(array[N-1] == (N-2)*(N-1)*0.5)  cout << "Check Passed " <<endl;
    else cout << " CHECK **FAILED** "<<endl;


    // Write to the File
    ofstream outfile;
    outfile.open("par.txt");

    for ( int k = 0 ; k < N ; k++)
        outfile << array[k] <<endl;
    
    outfile.close();

    // Free the GPU and Host Memory
    cudaFree(arrayDevice);
    delete[] array;

}

