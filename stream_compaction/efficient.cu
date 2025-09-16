#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int maxBlockSize = 128;

        // assumes padding to block size;
        __global__ void kernBlockScan(int n, int blockLog2Ceil, int* data) {
            int blockOffset = blockDim.x * blockIdx.x;
            int idx = threadIdx.x + 1;
            // upsweep
            for (int i = 1; i <= blockLog2Ceil; i++) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    data[pos] = data[pos] + data[pos - offset];
                }
                __syncthreads();
            }
            // downsweep
            // set root to 0

            if (threadIdx.x == blockDim.x - 1) {
                data[blockOffset + blockDim.x - 1] = 0;
            }

            for (int i = blockLog2Ceil; i > 0; i--) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    int t = data[pos - offset];
                    data[pos - offset] = data[pos];
                    data[pos] = t + data[pos];
                }

                __syncthreads();
            }
        }

        // assumes padding to block size;
        __global__ void kernBlockScanStoreSum(int n, int blockLog2Ceil, int* data, int* sums) {
            int blockOffset = blockDim.x * blockIdx.x;
            int idx = threadIdx.x + 1;
            // upsweep
            for (int i = 1; i <= blockLog2Ceil; i++) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    data[pos] = data[pos] + data[pos - offset];
                }
                __syncthreads();
            }
            // store sum for cross block sum later
            if (threadIdx.x == 0) {
                sums[blockIdx.x] = data[blockOffset + blockDim.x - 1];
            }
            __syncthreads();
            // downsweep
            // set root to 0

            if (threadIdx.x == blockDim.x - 1) {
                data[blockOffset + blockDim.x - 1] = 0;
            }

            for (int i = blockLog2Ceil; i > 0; i--) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    int t = data[pos - offset];
                    data[pos - offset] = data[pos];
                    data[pos] = t + data[pos];
                }

                __syncthreads();
            }
        }

        // assumes padding to block size;
        __global__ void kernAddSums(int* data, const int* sums) {
            int idx = blockDim.x * (blockIdx.x) + threadIdx.x;
            data[idx] += sums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            timer().startGpuTimer();

            //std::vector<int*> scanArrays{};

            //int scanArrayLen = divup(n, maxBlockSize) * maxBlockSize;
            //while (scanArrayLen > maxBlockSize) {
            //    int* d_array;
            //    cudaMalloc(&d_array, scanArrayLen * sizeof(int));
            //    checkCUDAError("cudaMalloc scanArrayLen failed");
            //    cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
            //    scanArrays.push_back(d_array);
            //    scanArrayLen = divup(n, scanArrayLen) * maxBlockSize;
            //}
            //{
            //    // scanArrayLen = maxBlockSize now
            //    int* d_array;
            //    cudaMalloc(&d_array, scanArrayLen * sizeof(int));
            //    checkCUDAError("cudaMalloc scanArrayLen failed");
            //    cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
            //    scanArrays.push_back(d_array);
            //}

            size_t sizeInBytes = n * sizeof(int);

            int* d_data;
            cudaMalloc(&d_data, sizeInBytes);
            checkCUDAError("cudaMalloc d_data failed");

            int blockSize = 4;
            int gridSize = divup(n, blockSize);

            int* d_sums;
            cudaMalloc(&d_sums, gridSize * sizeof(int));

            cudaMemcpy(d_data, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed");


            kernBlockScanStoreSum << <gridSize, blockSize >> > (n, ilog2ceil(blockSize), d_data, d_sums);
            kernBlockScan<< <1, gridSize >> > (n, ilog2ceil(gridSize), d_sums);
            kernAddSums << <gridSize, blockSize >> >(d_data, d_sums);

            // due to the fact that we want to make the inclusive scan exclusive instead
            //odata[0] = 0;
            //cudaMemcpy(odata + 1, d_data, sizeInBytes - sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
            //cudaMemcpy(odata, d_sums, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");

            cudaFree(d_data);
            checkCUDAError("cudaFree d_data failed");
            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
