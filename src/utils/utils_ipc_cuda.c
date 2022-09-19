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

#include "utils_ipc.h"
#include <stdio.h>

bool isCudaAddress(void *addr, bool *is_cuda_addr_ret, char *error_message, int max_error_message_chars) {
  struct cudaPointerAttributes attributes;
  cudaError_t cuda_status = cudaPointerGetAttributes(&attributes, addr);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to determine if memory addr is CUDA or CPU, operation may not be supported on this platform: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  
  *is_cuda_addr_ret = (attributes.type == cudaMemoryTypeDevice);
  return true;
}

// The creator of the memory needs to call this before passing (via some other IPC) it to the remote endpoint
bool cudaCreateIpcMapFromLocalAddr(void *cuda_addr, cudaIpcMemHandle_t *ipc_map_ret, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaIpcGetMemHandle(ipc_map_ret, cuda_addr);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to create a map to the CUDA address: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
  // NOTE: Does not appear to be a function to un'Get', so no need for a complimenting destroy function
}

// The remote endpoint calls this
void *cudaGetRemoteAddrFromIpcMap(cudaIpcMemHandle_t remote_ipc_map, char *error_message, int max_error_message_chars) {
  void *remote_cuda_addr = NULL;
  unsigned int flags = cudaIpcMemLazyEnablePeerAccess; // IMPORTANT: cudaIpcMemLazyEnablePeerAccess is required and looks to be the only option
  cudaError_t cuda_status = cudaIpcOpenMemHandle(&remote_cuda_addr, remote_ipc_map, flags);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to get a remote CUDA address from an IPC mapping: %s", cudaGetErrorString(cuda_status));
    return NULL;
  }
  return remote_cuda_addr;
}

// The remote endpoint calls this
bool cudaReleaseRemoteIpcMappedAddr(void *mapped_cuda_addr, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaIpcCloseMemHandle(mapped_cuda_addr);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to release the remote CUDA mapped address: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}

bool cudaEventAlloc(cudaEvent_t *event_ret, char *error_message, int max_error_message_chars) {
  // Allocate the CUDA event
  cudaError_t cuda_status = cudaEventCreateWithFlags(event_ret, cudaEventDisableTiming | cudaEventInterprocess);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to allocate CUDA event: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}

bool cudaEventFree(cudaEvent_t *event, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaEventDestroy(*event);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to free CUDA event: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}

// The creator of the event needs to call this before passing (via some other IPC) it to the remote endpoint
bool cudaCreateIpcMapFromLocalEvent(cudaEvent_t *event, cudaIpcEventHandle_t *ipc_map_ret, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaIpcGetEventHandle(ipc_map_ret, *event);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to create a map to the CUDA event: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
  // NOTE: Does not appear to be a function to un'Get', so no need for a complimenting destroy function
}

// The remote endpoint calls this
bool cudaGetRemoteEventFromIpcMap(cudaIpcEventHandle_t remote_ipc_map, cudaEvent_t *remote_event_ret, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaIpcOpenEventHandle(remote_event_ret, remote_ipc_map);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to get a remote CUDA event from an IPC mapping: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}

// Either endpoint can call this, but really meant for the sender to call after the cuda memcpy is complete
bool cudaEventNotify(cudaEvent_t *event, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaEventRecord(*event, 0);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to mark CUDA event as notified: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}

// Either endpoint can call this, but really meant for the sender to verify a previous use of the event is still not active
bool cudaEventAvailable(cudaEvent_t *event, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaEventQuery(*event);
  if (cuda_status != cudaSuccess) {
    if (cuda_status == cudaErrorNotReady) {
      snprintf(error_message, max_error_message_chars, "Previous use of CUDA event is still not ready; i.e. a previous transfer is still not complete. May need to increase MAX_CUDA_EVENTS in interconnect_InterProcess.c");
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to wait for notification from CUDA event: %s", cudaGetErrorString(cuda_status));
    }
    return false;
  }
  return true;
}

// Either endpoint can call this, but really meant for the recever to call to block waiting for the sender to complete its transfer
bool cudaEventWait(cudaEvent_t *event, char *error_message, int max_error_message_chars) {
  cudaError_t cuda_status = cudaEventSynchronize(*event);
  if (cuda_status != cudaSuccess) {
    snprintf(error_message, max_error_message_chars, "Failed to wait for notification from CUDA event: %s", cudaGetErrorString(cuda_status));
    return false;
  }
  return true;
}
