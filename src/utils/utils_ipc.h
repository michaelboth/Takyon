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

#ifndef _utils_ipc_h_
#define _utils_ipc_h_

#include <stdbool.h>
#include <stdint.h>
#ifdef ENABLE_CUDA
  #include "cuda_runtime.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// Memory mapped allocation (can be shared between processes)
extern bool mmapAlloc(const char *map_name, uint64_t bytes, void **addr_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars);
extern bool mmapGet(const char *map_name, uint64_t bytes, void **addr_ret, bool *got_it_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars);
extern bool mmapFree(void *mmap_handle, char *error_message, int max_error_message_chars); // NOTE: call on both the local and remote mmap handles

#ifdef ENABLE_CUDA
// Check if local or remote addr is CUDA or CPU
extern bool isCudaAddress(void *addr, bool *is_cuda_addr_ret, char *error_message, int max_error_message_chars);

// Sharing a CUDA address
extern bool cudaCreateIpcMapFromLocalAddr(void *cuda_addr, cudaIpcMemHandle_t *ipc_map_ret, char *error_message, int max_error_message_chars);
extern void *cudaGetRemoteAddrFromIpcMap(cudaIpcMemHandle_t remote_ipc_map, char *error_message, int max_error_message_chars);
extern bool cudaReleaseRemoteIpcMappedAddr(void *mapped_cuda_addr, char *error_message, int max_error_message_chars);

// CUDA events need to synchronize cudaMemcpy() between processes
extern bool cudaEventAlloc(cudaEvent_t *event_ret, char *error_message, int max_error_message_chars);
extern bool cudaEventFree(cudaEvent_t *event, char *error_message, int max_error_message_chars); // NOTE: do not call this on the remote mapped event
extern bool cudaCreateIpcMapFromLocalEvent(cudaEvent_t *event, cudaIpcEventHandle_t *ipc_map_ret, char *error_message, int max_error_message_chars);
extern bool cudaGetRemoteEventFromIpcMap(cudaIpcEventHandle_t remote_ipc_map, cudaEvent_t *remote_event_ret, char *error_message, int max_error_message_chars);
extern bool cudaEventNotify(cudaEvent_t *event, char *error_message, int max_error_message_chars);
extern bool cudaEventWait(cudaEvent_t *event, char *error_message, int max_error_message_chars);
#endif

#ifdef __cplusplus
}
#endif

#endif
