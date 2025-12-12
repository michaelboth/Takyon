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

//#define PRINT_PROVIDERS

double Common::clockTimeSeconds() {
  struct timespec curr_time;
  clock_gettime(CLOCK_MONOTONIC, &curr_time); // CLOCK_MONOTONIC is only increasing
  long long total_nanoseconds = ((long long)curr_time.tv_sec * 1000000000LL) + (long long)curr_time.tv_nsec;
  return total_nanoseconds/1000000000.0;
}

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

void* Common::allocateTransportMemory(uint64_t _bytes, bool _is_for_rdma) { /*+ is_for_gpu */
  (void)_is_for_rdma; // Quiets compiler if ENABLE_CUDA is not defined
  void *addr = NULL;
#ifdef ENABLE_CUDA
  cudaError_t cuda_status = cudaMalloc(&addr, _bytes);
  if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
  if (_is_for_rdma) {
    // Since this memory will transfer asynchronously via GPUDirect, need to mark the memory to be synchronous when accessing it after being received
    unsigned int flag = 1;
    CUresult cuda_result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)addr);
    if (cuda_result != CUDA_SUCCESS) { printf("cuPointerSetAttribute() failed: return_code=%d\n", cuda_result); exit(EXIT_FAILURE); }
  }
#else
  addr = malloc(_bytes);
#endif
  return addr;
}

void Common::freeTransportMemory(void *_addr) {
#ifdef ENABLE_CUDA
  cudaFree(_addr);
#else
  free(_addr);
#endif
}
