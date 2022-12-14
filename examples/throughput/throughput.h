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

#ifndef _throughput_h_
#define _throughput_h_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern void throughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t src_buffer_count, const uint32_t dest_buffer_count, const bool use_polling_completion, const char *transfer_mode, const bool validate);

#ifdef __cplusplus
}
#endif

#endif
