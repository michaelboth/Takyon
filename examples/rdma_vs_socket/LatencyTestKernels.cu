//     Copyright 2025 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

#include "LatencyTestKernels.hpp"
#include "Common.hpp"
#include "cuda_runtime.h"
#include "math.h"

#define IDEAL_1D_BLOCK_SIZE 256

static uint32_t *L_result_buffer = NULL;

void LatencyTestKernels::init() {
  // Create temp cuda memory to hold sumation result
  cudaError_t cuda_status = cudaMalloc(&L_result_buffer, sizeof(uint32_t));
  if (cuda_status != cudaSuccess) {
    EXIT_WITH_MESSAGE(std::string("cudaMalloc() failed: return_code=" + std::to_string(cuda_status)));
  }
}

void LatencyTestKernels::finalize() {
  // Free the result memory
  cudaError_t cuda_status = cudaFree(L_result_buffer);
  if (cuda_status != cudaSuccess) {
    EXIT_WITH_MESSAGE(std::string("cudaFree() failed: return_code=" + std::to_string(cuda_status)));
  }
  L_result_buffer = NULL;
}

static __global__ void fillInValidationDataKernel(uint32_t *_buffer, uint32_t _count, uint32_t _starting_value) {
  // compute the x index and validate
  int index = threadIdx.x + (blockDim.x * blockIdx.x);
  if (index >= _count) return;
  _buffer[index] = _starting_value + index;
}

static __global__ void copyKernel(uint32_t *_input_buffer, uint32_t *_output_buffer, uint32_t _count) {
  // compute the x index and validate
  int index = threadIdx.x + (blockDim.x * blockIdx.x);
  if (index >= _count) return;
  _output_buffer[index] = _input_buffer[index];
}

static __inline__ __device__ uint32_t warpReduceSum(uint32_t _value) {
  // Synchronously sum up the values in the warp registers, without the need to touch shared memory or global memory
  for (int width=16; width>0; width/=2) {
    _value += __shfl_down_sync(0xffffffff, _value, width);
  }
  return _value;
}

static __global__ void validateDataKernel(uint32_t* __restrict__ _input_buffer, uint32_t _count, uint32_t _starting_value, uint32_t* __restrict__ _result_buffer) {
  // Setup the shared memory
  extern __shared__ uint32_t shared_mem_buffer[]; // This memory is only visible to the current block. Each block has it's own memory

  // Determine current thread index and value index
  int thread_index = threadIdx.x;
  int index = blockIdx.x * blockDim.x + thread_index;

  // Determine is the value is invalid or not
  uint32_t thread_sum = 0;
  if (index < _count) {
    uint32_t expected_value = _starting_value + index;
    if (_input_buffer[index] != expected_value) {
      // Value is invalid
      thread_sum++;
    }
  }

  // Warp reduction
  thread_sum = warpReduceSum(thread_sum);

  // Store warp sums to shared memory (only use the first thread of each warp)
  if ((thread_index % 32) == 0) {
    shared_mem_buffer[thread_index/32] = thread_sum;
  }
  // Make sure all warps in the block are synchronized
  __syncthreads();

  // First warp does final reduction
  if (thread_index < 32) {
    int max_shared_buffer_index = (blockDim.x + 31) / 32;
    uint32_t sum = (thread_index < max_shared_buffer_index) ? shared_mem_buffer[thread_index] : 0; // Use zero if outside the shared_mem_buffer
    uint32_t block_sum = warpReduceSum(sum);
    if (thread_index == 0) {
      // Each thread of each block will atomically add to the single result buffer
      // NOTE: This is not efficient since many blocks will call this, but this is for validation and does not need to be efficient
      atomicAdd(_result_buffer, block_sum);
    }
  }
}

void LatencyTestKernels::runFillInValidationDataKernelBlocking(uint32_t *_buffer, uint64_t _count, uint32_t _starting_value) {
  // Prepare values for kernel
  int block_count = (int)ceil(_count / (double)IDEAL_1D_BLOCK_SIZE);
  dim3 block_sizes(IDEAL_1D_BLOCK_SIZE,1,1);
  dim3 block_counts(block_count,1,1);
  size_t shared_bytes_per_block = 0;
  cudaStream_t stream = 0; // The default global stream

  // Submit the kernel
  fillInValidationDataKernel<<<block_counts, block_sizes, shared_bytes_per_block, stream>>>(_buffer, (uint32_t)_count, _starting_value);

  // Wait for the kernel to complete
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    EXIT_WITH_MESSAGE(std::string("cudaStreamAttachMemAsync() failed: return_code=" + std::to_string(cuda_status)));
  }
}

void LatencyTestKernels::runCopyKernelBlocking(uint32_t *_input_buffer, uint32_t *_output_buffer, uint64_t _count) {
  // Prepare values for kernel
  int block_count = (int)ceil(_count / (double)IDEAL_1D_BLOCK_SIZE);
  dim3 block_sizes(IDEAL_1D_BLOCK_SIZE,1,1);
  dim3 block_counts(block_count,1,1);
  size_t shared_bytes_per_block = 0;
  cudaStream_t stream = 0; // The default global stream

  // Submit the kernel
  copyKernel<<<block_counts, block_sizes, shared_bytes_per_block, stream>>>(_input_buffer, _output_buffer, (uint32_t)_count);

  // Wait for the kernel to complete
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    EXIT_WITH_MESSAGE(std::string("cudaStreamAttachMemAsync() failed: return_code=" + std::to_string(cuda_status)));
  }
}

uint32_t LatencyTestKernels::runValidateDataKernelBlocking(uint32_t *_buffer, uint64_t _count, uint32_t _starting_value) {
  // Prepare values for kernel
  int block_count = (int)ceil(_count / (double)IDEAL_1D_BLOCK_SIZE);
  dim3 block_sizes(IDEAL_1D_BLOCK_SIZE,1,1);
  dim3 block_counts(block_count,1,1);
  size_t shared_bytes_per_block = (IDEAL_1D_BLOCK_SIZE / 32) * sizeof(uint32_t);
  cudaStream_t stream = 0; // The default global stream

  // Submit the kernel
  validateDataKernel<<<block_counts, block_sizes, shared_bytes_per_block, stream>>>(_buffer, (uint32_t)_count, _starting_value, L_result_buffer);

  // Wait for the kernel to complete
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    EXIT_WITH_MESSAGE(std::string("cudaStreamAttachMemAsync() failed: return_code=" + std::to_string(cuda_status)));
  }

  // Copy result to host
  uint32_t result = 0;
  cudaMemcpy(&result, L_result_buffer, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  return result;
}
