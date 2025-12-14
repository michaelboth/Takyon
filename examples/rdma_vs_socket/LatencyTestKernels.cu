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
