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

#include "Common.hpp"
#include <fstream>
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <Windows.h>
  #include <sys/timeb.h>
#endif
#ifdef ENABLE_CUDA
  #ifndef _WIN32
    #include "cuda.h"
  #endif
  #include "cuda_runtime.h"
#endif

//#define PRINT_PROVIDERS

#ifdef ENABLE_CUDA
  #define CUDA_DEVICE_ID 0
#endif

#if defined(_WIN32)
double Common::clockTimeSeconds() {
  static long long base_time;
  static int got_base_time = 0;
  static int tested_freq = 0;
  static long long freq = 0L;
  if (!tested_freq) {
    tested_freq = 1;
    if (!QueryPerformanceFrequency((LARGE_INTEGER *)&freq)) {
      // High speed clock not available
      freq = 0L;
    }
  }
  int got_high_speed_clock = 0;
  long long total_nanoseconds;
  if (freq > 0L) {
    // Uses high performance clock, with a frequency of 'freq'
    long long count;
    if (QueryPerformanceCounter((LARGE_INTEGER *)&count)) {
      long long seconds;
      long long nanoseconds;
      long long nanoseconds_per_second = 1000000000L;
      got_high_speed_clock = 1;
      seconds = count / freq;
      count = count % freq;
      nanoseconds = (nanoseconds_per_second * count) / freq;
      total_nanoseconds = (seconds * nanoseconds_per_second) + nanoseconds;
    } else {
      // The high frequency clock may have stopped working mid run, or right from the beginning
      freq = 0L;
      got_high_speed_clock = 0;
    }
  }
  if (!got_high_speed_clock) {
    // Uses the low resolution wall clock
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    total_nanoseconds = (long long)timebuffer.time * (long long)1000000000;
    total_nanoseconds += ((long long)(timebuffer.millitm) * (long long)(1000000));
  }
  if (!got_base_time) {
    base_time = total_nanoseconds;
    got_base_time = 1;
  }
  return (total_nanoseconds - base_time)/1000000000.0;
}

#else

double Common::clockTimeSeconds() {
  struct timespec curr_time;
  clock_gettime(CLOCK_MONOTONIC, &curr_time); // CLOCK_MONOTONIC is only increasing
  long long total_nanoseconds = ((long long)curr_time.tv_sec * 1000000000LL) + (long long)curr_time.tv_nsec;
  return total_nanoseconds/1000000000.0;
}
#endif

double Common::smoothValue(double _old_value, double _new_value, double _new_factor) {
  return _new_value * _new_factor + _old_value * (1.0 - _new_factor);
}

std::map<std::string, std::string> Common::loadProviderParamsFile(std::string _filename) {
  // Open file
  std::ifstream infile(_filename);
  if (!infile.is_open()) {
    EXIT_WITH_MESSAGE(std::string("Failed to open file: '" + _filename + "'"));
  }

  // Load params
  std::map<std::string, std::string> params;
  int line_number = 0;
  while (!infile.eof()) {
    // Read the line
    std::string line;
    std::getline(infile, line);
    line_number++;
    // Ignore comments
    if (line.starts_with("#")) line = "";
    // Store params
    if (line.size() > 0) {
      if (line.starts_with("TCP_A: ")) { params["TCP_A"] = line.substr(7); }
      else if (line.starts_with("TCP_B: ")) { params["TCP_B"] = line.substr(7); }
      else if (line.starts_with("UDP_A: ")) { params["UDP_A"] = line.substr(7); }
      else if (line.starts_with("UDP_B: ")) { params["UDP_B"] = line.substr(7); }
      else if (line.starts_with("RC_A: ")) { params["RC_A"] = line.substr(6); }
      else if (line.starts_with("RC_B: ")) { params["RC_B"] = line.substr(6); }
      else if (line.starts_with("UC_A: ")) { params["UC_A"] = line.substr(6); }
      else if (line.starts_with("UC_B: ")) { params["UC_B"] = line.substr(6); }
      else if (line.starts_with("UD_A: ")) { params["UD_A"] = line.substr(6); }
      else if (line.starts_with("UD_B: ")) { params["UD_B"] = line.substr(6); }
      else {
        EXIT_WITH_MESSAGE(std::string("Unknown param at line " + std::to_string(line_number) + ": '" + line + "'"));
      }
    }
  }

  // Close file
  infile.close();
  if (params.size() != 10) {
    EXIT_WITH_MESSAGE(std::string("Params file: '" + _filename + "' does not have all connection variations: TCP, UDP, RC, UC, and UD"));
  }

#ifdef PRINT_PROVIDERS
  printf("Providers:\n");
  for (const auto &pair: params) {
    printf("  Key: '%s',  Value: '%s'\n", pair.first.c_str(), pair.second.c_str());
  }
  printf("\n");
#endif

  return params;
}

Common::MemoryType Common::memoryTypeToUseForTransport(bool _is_for_rdma, bool _is_for_gpu) {
  if (_is_for_gpu) {
#ifdef ENABLE_CUDA
    // Set the CUDA device id
    int device_id = CUDA_DEVICE_ID;
    cudaSetDevice(device_id);

    // See if it's an integrated GPU
    cudaDeviceProp properties;
    cudaError_t cuda_status = cudaGetDeviceProperties(&properties, device_id);
    if (cuda_status != cudaSuccess) {
      EXIT_WITH_MESSAGE(std::string("cudaGetDeviceProperties() failed: return_code=" + std::to_string(cuda_status)));
    }
    if (properties.integrated) {
      printf("Integrated GPU: '%s', Compute Capability: %d.%d:  Transfers and processing will be in memory shared by the CPU and GPU (no copying between CPU and GPU).\n", properties.name, properties.major, properties.minor);
      return Common::MemoryType::SocIntegratedGPU;
    }

    // If not RDMA then avoid GPUDirect
    if (!_is_for_rdma) {
      printf("Discrete GPU: '%s', Compute Capability: %d.%d:  Not using GPUDirect, so transfers will be copied from CPU to GPU.\n", properties.name, properties.major, properties.minor);
      return Common::MemoryType::DiscreteGPU_withoutGPUDirect;
    }

    // See if device supports GPUDirect
    // Get a handle to the device
    CUdevice cuda_dev_handle;
    CUresult cuda_result;
    cuda_result = cuDeviceGet(&cuda_dev_handle, device_id);
    if (cuda_result != CUDA_SUCCESS) {
      EXIT_WITH_MESSAGE(std::string("cuDeviceGet() failed: return_code=" + std::to_string(cuda_result)));
    }

    // See if GPUDirect is supported
    int supports_gpudirect_result = 0;
    cuda_result = cuDeviceGetAttribute(&supports_gpudirect_result, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cuda_dev_handle);
    if (cuda_result != CUDA_SUCCESS) {
      EXIT_WITH_MESSAGE(std::string("cuDeviceGetAttribute() failed: return_code=" + std::to_string(cuda_result)));
    }
    if (supports_gpudirect_result == 1) {
      printf("Discrete GPU: '%s', Compute Capability: %d.%d:  Will use GPUDirect, so transfers will be direct to GPU.\n", properties.name, properties.major, properties.minor);
      return Common::MemoryType::DiscreteGPU_withGPUDirect;
    }

    // Falling back to discrete GPU with no GPUDirect
    printf("Discrete GPU: '%s', Compute Capability: %d.%d:  GPUDirect not available, so transfers will be copied from CPU to GPU.\n", properties.name, properties.major, properties.minor);
    return Common::MemoryType::DiscreteGPU_withGPUDirect;
#else
    (void)_is_for_rdma; // Quiets compiler if ENABLE_CUDA is not defined
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
#endif
  }

  printf("Transfers and processing will be in CPU memory\n");
  return Common::MemoryType::CPU;
}

void* Common::allocateTransportMemory(uint64_t _bytes, Common::MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    return malloc(_bytes);
  }

#ifdef ENABLE_CUDA
  if (_memory_type == Common::MemoryType::SocIntegratedGPU) {
    void *addr = NULL;
    unsigned int flags = 0;
    cudaError_t cuda_status = cudaHostAlloc(&addr, _bytes, flags); // Allocates host memory and pins it so can be accessed by the GPU directly (SoC integrated GPU allow this)
    if (cuda_status != cudaSuccess) {
      EXIT_WITH_MESSAGE(std::string("cudaHostAlloc() failed: return_code=" + std::to_string(cuda_status)));
    }
    return addr;

  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect || _memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    void *addr = NULL;
    cudaError_t cuda_status = cudaMalloc(&addr, _bytes);
    if (cuda_status != cudaSuccess) {
      EXIT_WITH_MESSAGE(std::string("cudaMalloc() failed: return_code=" + std::to_string(cuda_status)));
    }
    if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
      // Since this memory will transfer asynchronously via GPUDirect, need to mark the memory to be synchronous when accessing it after being received
      unsigned int flag = 1;
      CUresult cuda_result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)addr);
      if (cuda_result != CUDA_SUCCESS) {
        EXIT_WITH_MESSAGE(std::string("cuPointerSetAttribute() failed: return_code=" + std::to_string(cuda_result)));
      }
    }
    return addr;

  } else {
    EXIT_WITH_MESSAGE(std::string("Memory type unknown"));
  }

#else
  EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
#endif
  return NULL;
}

void Common::freeTransportMemory(void *_addr, MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    free(_addr);
    return;
  }
#ifdef ENABLE_CUDA
  if (_memory_type == Common::MemoryType::SocIntegratedGPU) {
    cudaError_t cuda_status = cudaFreeHost(_addr);
    if (cuda_status != cudaSuccess) {
      EXIT_WITH_MESSAGE(std::string("cudaFreeHost() failed: return_code=" + std::to_string(cuda_status)));
    }
    return;
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect || _memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    cudaError_t cuda_status = cudaFree(_addr);
    if (cuda_status != cudaSuccess) {
      EXIT_WITH_MESSAGE(std::string("cudaFree() failed: return_code=" + std::to_string(cuda_status)));
    }
    return;
  } else {
    EXIT_WITH_MESSAGE(std::string("Memory type unknown"));
  }
#else
  EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
#endif
}

std::string Common::memoryTypeToText(MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) return "CPU";
  if (_memory_type == Common::MemoryType::SocIntegratedGPU) return "IntegratedGPU";
  if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) return "DiscreteGPU_withGPUDirect";
  if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) return "DiscreteGPU_withoutGPUDirect";
  return "unknown";
}
