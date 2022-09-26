// Takyon 1.x was originally developed by Michael Both at Abaco, and he is now continuing development independently
//
// Original copyright:
//     Copyright 2018,2020 Abaco Systems
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.
//
// Changes for 2.0 (starting from Takyon 1.1.0):
//   - See comments in takyon.h for the bigger picture of the changes
//   - Minor changes in this file
//
// Copyright for modifications:
//     Copyright 2022 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

#include "utils_endian.h"

bool endianIsBig() {
  long int testInt = 0x12345678;
  char *ptr = (char *)&testInt;
  if (ptr[0] == 0x78) {
    return false;
  } else {
    return true;
  }
}

void endianSwap2Byte(void *data, uint64_t num_elements) {
  uint16_t *data2 = (uint16_t *)data;
  for (uint64_t i=0; i<num_elements; i++) {
    uint16_t value = data2[i];
    data2[i] = (uint16_t)((value>>8) | (value<<8));
  }
}

void endianSwap4Byte(void *data, uint64_t num_elements) {
  uint32_t *data2 = (uint32_t *)data;
  for (uint64_t i=0; i<num_elements; i++) {
    uint32_t value = data2[i];
    value = ((value << 8) & 0xFF00FF00) | ((value >> 8) & 0xFF00FF);
    data2[i] = (value << 16) | (value >> 16);
  }
}

void endianSwap8Byte(void *data, uint64_t num_elements) {
  uint64_t *data2 = (uint64_t *)data;
  for (uint64_t i=0; i<num_elements; i++) {
    uint64_t value = data2[i];
    value = ((value << 8) & 0xFF00FF00FF00FF00ULL ) | ((value >> 8) & 0x00FF00FF00FF00FFULL );
    value = ((value << 16) & 0xFFFF0000FFFF0000ULL ) | ((value >> 16) & 0x0000FFFF0000FFFFULL );
    data2[i] = (value << 32) | (value >> 32);
  }
}
