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

#pragma once

#include <string>
#include <map>
#include <cstdint>

#define EXIT_WITH_MESSAGE(_m) do { printf("ERROR in '%s:%s()' line %d: %s\n", __FILE__, __FUNCTION__, __LINE__, _m.c_str()); exit(EXIT_FAILURE); } while (false)

#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
  #define UINT64_FORMAT_7CHARS "%7llu"
#else
  #define UINT64_FORMAT "%ju"
  #define UINT64_FORMAT_7CHARS "%7ju"
#endif

namespace Common {
  constexpr uint64_t MIN_NBYTES = 4;
  constexpr uint64_t MAX_NBYTES = (4*1024*1024);
  constexpr uint32_t DEFAULT_NITERS = 100;
  constexpr uint32_t DEFAULT_NBUFS = 10;    // Ignored for latency test
  constexpr double ELAPSED_SECONDS_TO_PRINT = 0.2;

  enum class MemoryType {
    Unset,
    CPU,                           // Transfers and validation are done on CPU
    SocIntegratedGPU,              // Should be able to run CUDA kernels on host allocated memory; use cudaHostAlloc() as suggested by NVIDIA engineers
    DiscreteGPU_withoutGPUDirect,  // Will need to copy transfers from CPU to GPU
    DiscreteGPU_withGPUDirect,     // Zero-copy transfers
  };

  struct AppParams {
    std::string provider;
    std::string provider_params;
    uint32_t nbufs = DEFAULT_NBUFS;
    bool run_forever = true;
    uint32_t iters = DEFAULT_NITERS;
    uint64_t nbytes = 0; // If 0, will loop from MIN_NBYTES to MAX_NBYTES
    bool use_polling = false;
    bool is_for_gpu = false;
    bool validate = false;
    bool verbose = false;
    void *unikorn_session = NULL;
  };

  double clockTimeSeconds();
  double smoothValue(double _old_value, double _new_value, double _new_factor);
  std::map<std::string, std::string> loadProviderParamsFile(std::string _filename);
  MemoryType memoryTypeToUseForTransport(bool _is_for_rdma, bool _is_for_gpu);
  void* allocateTransportMemory(uint64_t _bytes, MemoryType _memory_type);
  void freeTransportMemory(void *_addr, MemoryType _memory_type);
  std::string memoryTypeToText(MemoryType _memory_type);
};
