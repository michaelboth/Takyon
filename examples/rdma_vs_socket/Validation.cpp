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

#include "Validation.hpp"
#ifdef ENABLE_CUDA
  #include "ValidationKernels.hpp"
#endif

void Validation::fillInData(uint32_t *_buffer, uint32_t *_buffer_gpu, uint64_t _count, uint64_t _starting_value, Common::MemoryType _memory_type) {
  (void)_buffer_gpu; // Used to quiet the compile if ENABLE_CUDA is not defined
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) { _buffer[i] = (uint32_t)(_starting_value+i); }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU || _memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
    ValidationKernels::runFillInValidationDataKernelBlocking(_buffer, _count, _starting_value);
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    ValidationKernels::runFillInValidationDataKernelBlocking(_buffer_gpu, _count, _starting_value);
    Common::gpuToHostCopy(_buffer, _buffer_gpu, _count*sizeof(uint32_t));
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}

void Validation::copyData(uint32_t *_input_buffer, uint32_t *_output_buffer, uint64_t _count, Common::MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) { _output_buffer[i] = _input_buffer[i]; }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU || _memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect || _memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    ValidationKernels::runCopyKernelBlocking(_input_buffer, _output_buffer, _count);
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}

void Validation::validateData(uint32_t *_buffer, uint32_t *_buffer_gpu, uint64_t _count, uint64_t _starting_value, Common::MemoryType _memory_type) {
  (void)_buffer_gpu; // Used to quiet the compile if ENABLE_CUDA is not defined
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) {
      uint32_t expected_value = (uint32_t)(_starting_value+i);
      if (_buffer[i] != expected_value) {
        EXIT_WITH_MESSAGE(std::string("Received invalid data, _starting_value=" + std::to_string(_starting_value) + ", at index " + std::to_string(i) + " expected  " + std::to_string(expected_value) + " but got " + std::to_string(_buffer[i])));
      }
    }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU || _memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
    uint32_t invalid_value_count = ValidationKernels::runValidateDataKernelBlocking(_buffer, _count, _starting_value);
    if (invalid_value_count > 0) {
      EXIT_WITH_MESSAGE(std::string("Received " + std::to_string(invalid_value_count) + " invalid values"));
    }
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    Common::hostToGpuCopy(_buffer_gpu, _buffer, _count*sizeof(uint32_t));
    uint32_t invalid_value_count = ValidationKernels::runValidateDataKernelBlocking(_buffer_gpu, _count, _starting_value);
    if (invalid_value_count > 0) {
      EXIT_WITH_MESSAGE(std::string("Received " + std::to_string(invalid_value_count) + " invalid values"));
    }
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}
