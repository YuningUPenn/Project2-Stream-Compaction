#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScan(int N, int* odata, const int* idata, int pow) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= N) {
                return;
            }

            int diff = 1 << pow;
            if (index >= diff) {
                odata[index] = idata[index - diff] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int* dev_arrA;
            int* dev_arrB;

            cudaMalloc((void**)&dev_arrA, n * sizeof(int));
            cudaMalloc((void**)&dev_arrB, n * sizeof(int));
            cudaMemcpy(dev_arrA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer(); // not to include any initial/final memory operations

            int depth = ilog2ceil(n);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            int* in;
            int* out;

            for (int i = 0; i < depth; i++) {
                if (i % 2 == 0) {
                    in = dev_arrA;
                    out = dev_arrB;
                }
                else {
                    in = dev_arrB;
                    out = dev_arrA;
                }
                naiveScan << <fullBlocksPerGrid, blockSize >> > (n, out, in, i);
            }

            timer().endGpuTimer(); // not to include any initial/final memory operations

            cudaMemcpy(odata, out, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            cudaFree(dev_arrA);
            cudaFree(dev_arrB);
            
        }
    }
}
