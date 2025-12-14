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

#include <cstdint>

namespace LatencyTestKernels {
  void runFillInValidationDataKernelBlocking(uint32_t *_buffer, uint64_t _count, uint32_t _starting_value);
  void runCopyKernelBlocking(uint32_t *_input_buffer, uint32_t *_output_buffer, uint64_t _count);
};
