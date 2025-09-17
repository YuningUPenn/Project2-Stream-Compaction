#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // up sweep function
        __global__ void upSweep(int N, int* data, int pow) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int end = (index + 1) * (1 << (pow + 1)) - 1;
            if (end >= N) {
                return;
            }
            int start = end - (1 << pow);
            data[end] += data[start];
        }

        // down sweep function
        __global__ void downSweep(int N, int* data, int pow) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int end = (index + 1) * (1 << (pow + 1)) - 1;
            if (end >= N) {
                return;
            }
            int start = end - (1 << pow);

            int temp = data[end];
            data[end] += data[start];
            data[start] = temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_tree;

            int depth = ilog2ceil(n);
            int N = 1 << depth;

            cudaMalloc((void**)&dev_tree, N * sizeof(int));
            cudaMemcpy(dev_tree, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer(); // not to include any initial/final memory operations

            // up sweep
            for (int d = 0; d < depth; d++) {
                dim3 curGrid((N / (1 << (d + 1)) + blockSize - 1) / blockSize);
                upSweep << <curGrid, blockSize >> > (N, dev_tree, d);
            }

            // down sweep
            //dev_tree[n - 1] = 0;
            cudaMemset(dev_tree + N - 1, 0, sizeof(int));
            for (int d = depth - 1; d >= 0; d--) {
                dim3 curGrid((N / (1 << (d + 1)) + blockSize - 1) / blockSize);
                downSweep << <curGrid, blockSize >> > (N, dev_tree, d);
            }

            timer().endGpuTimer(); // not to include any initial/final memory operations

            cudaMemcpy(odata, dev_tree, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_tree);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            
            //timer().startGpuTimer();£¿
            // TODO
            int* dev_idata;
            int* dev_odata;
            int* dev_indices;
            int* dev_bool;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bool, dev_idata);
            StreamCompaction::Efficient::scan(n, dev_indices, dev_bool);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bool, dev_indices);

            int count;
            int check;
            cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&check, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            cudaFree(dev_bool);
            //timer().endGpuTimer();£¿
            return count + check;
        }
    }
}
