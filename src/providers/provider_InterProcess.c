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
//   - Redesigned to the 2.x API functionality
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

#include "provider_InterProcess.h"
#include "takyon_private.h"
#include "utils_socket.h"
#include "utils_ipc.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#ifdef ENABLE_CUDA
  #include "cuda_runtime.h"
#endif
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

// Supported formats:
//   "InterProcessRC -pathID=<non_negative_integer>"
//   "InterProcessUC -pathID=<non_negative_integer>"

#define KLUDGE_USEC_SLEEP_BETWEEN_POLL_ATTEMPTS 100
#define MAX_CUDA_EVENTS 100  // Verify does not exceed attrs.max_pending_recv_requests

// This is use to allow for zero byte messages
#define MY_MAX(_a,_b) ((_a>_b) ? _a : _b)

// IMPORTANT: This is not stored, but is used during the init phase to get mmap address
typedef struct {
  uint64_t bytes;
  char name[TAKYON_MAX_BUFFER_NAME_CHARS]; // Only needed for special memory like inter-process mmaps. Ignore if not needed
#ifdef ENABLE_CUDA
  bool is_cuda;
  cudaIpcMemHandle_t ipc_addr_map;
  cudaIpcEventHandle_t ipc_event_map[MAX_CUDA_EVENTS];
#endif
} RemoteTakyonBufferInfo;

// Store this in buffer->private_data
typedef struct {
  TakyonPath *path;
#ifdef ENABLE_CUDA
  bool is_cuda;
  uint32_t curr_cuda_event_index;
  bool cuda_event_allocated[MAX_CUDA_EVENTS];
  cudaEvent_t cuda_event[MAX_CUDA_EVENTS];
#endif
} PrivateTakyonBuffer;

typedef struct {
  uint64_t bytes;
  void *mmap_addr;   // CPU or CUDA
  void *mmap_handle; // Only for CPU memory
#ifdef ENABLE_CUDA
  bool is_cuda;
  uint32_t curr_cuda_event_index;
  bool send_notification; // Used on the sender side if the memory buffer is used in the transfer
  cudaEvent_t cuda_event[MAX_CUDA_EVENTS];
#endif
} RemoteTakyonBuffer;

// IMPORTANT: A single recv request will contain one or more of these equal to the number of sub buffers,
//            and each request will use max_multi_blocks_per_recv_request elements
typedef struct {
  volatile bool transfer_posted;   // takyonPostRecvs() sets this to true and takyonSend() sets this to false, so no need to Mutex protect
  // Results
  volatile bool transfer_complete; // takyonPostRecvs() sets this false and takyonSend() sets this to true, so no need to Mutex protect
  uint64_t bytes_received;         // IMPORTANT: Set this before transfer_complete is set to true
  uint32_t piggyback_message;      // IMPORTANT: Set this before transfer_complete is set to true
  // Overall request. IMPORTANT: set all this before transfer_posted is set to true
  uint32_t sub_buffer_count;
  // Sub buffer. IMPORTANT: set all these before transfer_posted is set to true
  uint32_t buffer_index;
  uint64_t bytes;                  // Receiver can make this more than what is actually sent, takyonIsRecved() will report that actual bytes received
  uint64_t offset;                 // In bytes
} RecvRequestAndCompletion;

typedef struct {
  uint32_t max_pending_recv_requests;
  uint32_t max_sub_buffers_per_recv_request;
  char mmap_name[TAKYON_MAX_BUFFER_NAME_CHARS];
} RemotePathInfo;

typedef struct {
  bool is_unreliable;

  // Socket connection to init and finalize the path, also to detect disconnects
  TakyonSocket socket_fd;
  bool thread_started;
  pthread_t disconnect_detection_thread_id;
  bool connection_failed;

  // Remote buffers
  uint32_t remote_buffer_count;
  RemoteTakyonBuffer *remote_buffers;

  // Track posted recvs
  //   - Memory mapped circular buffer
  //   - List size == max_pending_recv_requests * max_multi_blocks_per_recv_request;
  //   - A single request will reserve max_multi_blocks_per_recv_request elements, but may use less
  uint32_t curr_local_unused_recv_request_index;
  uint32_t curr_local_posted_recv_request_index;
  RecvRequestAndCompletion *local_recv_request_and_completions;
  void *local_postings_mmap_handle;
  uint32_t max_remote_pending_recv_requests;
  uint32_t max_remote_sub_buffers_per_recv_request;
  uint32_t curr_remote_unused_recv_request_index;
  uint32_t curr_remote_posted_recv_request_index;
  RecvRequestAndCompletion *remote_recv_request_and_completions;
  void *remote_postings_mmap_handle;
} PrivateTakyonPath;

static void *disconnectDetectionThread(void *user_data) {
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)user_data;
  // Wait for either a socket disconnect, or for takyonDestroy() to get called
  uint32_t dummy;
  int64_t timeout_nano_seconds = -1; // Wait forever
  bool timed_out = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  if (!socketRecv(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
  }
  return NULL;
}

static bool freeResources(TakyonPath *path, PrivateTakyonPath *private_path, RemoteTakyonBufferInfo *remote_buffer_info_list, RemoteTakyonBufferInfo *local_buffer_info_list, bool report_errors) {
  bool success = true;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Connection was made, so disconnect gracefully
  // NOTE: TCP_NODELAY is likely already active, so just need to provide some time for the remote side to get any in-transit data before a disconnect message is sent
  // NOTE: private_path->connection_failed may be true, but still want to provide time for remote side to handle arriving data
  clockSleepYield(MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING);

  // Free mmaps and memory
  // Release path info
  if (private_path->remote_recv_request_and_completions != NULL) {
    // Release remote path info
    if (!mmapFree(private_path->remote_postings_mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      if (report_errors) TAKYON_RECORD_ERROR(path->error_message, "mmapFree() failed: %s\n", error_message);
      success = false;
    }
  }
  if (private_path->local_recv_request_and_completions != NULL) {
    // Release local path info
    if (!mmapFree(private_path->local_postings_mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      if (report_errors) TAKYON_RECORD_ERROR(path->error_message, "mmapFree() failed: %s\n", error_message);
      success = false;
    }
  }

  // Remote buffers
  for (uint32_t i=0; i<private_path->remote_buffer_count; i++) {
    RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[i];
    bool is_cuda = false;
#ifdef ENABLE_CUDA
    is_cuda = remote_buffer->is_cuda;
    if (is_cuda) {
      if (remote_buffer->mmap_addr != NULL) {
        if (!cudaReleaseRemoteIpcMappedAddr(remote_buffer->mmap_addr, error_message, MAX_ERROR_MESSAGE_CHARS)) {
          if (report_errors) TAKYON_RECORD_ERROR(path->error_message, "cudaReleaseRemoteIpcMappedAddr() failed: %s\n", error_message);
          success = false;
        }
      }
    }
#endif
    if (!is_cuda) {
      if (remote_buffer->mmap_addr != NULL) {
        if (!mmapFree(remote_buffer->mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
          if (report_errors) TAKYON_RECORD_ERROR(path->error_message, "mmapFree() failed: %s\n", error_message);
          success = false;
        }
      }
    }
  }

  if (private_path->remote_buffers != NULL) free(private_path->remote_buffers);
  if (remote_buffer_info_list != NULL) free(remote_buffer_info_list);
  if (local_buffer_info_list != NULL) free(local_buffer_info_list);

  // Local buffers
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    PrivateTakyonBuffer *private_buffer = (PrivateTakyonBuffer *)buffer->private_data;
    if (private_buffer != NULL) {
#ifdef ENABLE_CUDA
      if (private_buffer->is_cuda) {
        for (uint32_t j=0; j<MAX_CUDA_EVENTS; j++) {
          if (private_buffer->cuda_event_allocated[j] && !cudaEventFree(&private_buffer->cuda_event[j], error_message, MAX_ERROR_MESSAGE_CHARS)) {
            if (report_errors) TAKYON_RECORD_ERROR(path->error_message, "cudaEventFree() failed: %s\n", error_message);
            success = false;
          }
        }
      }
#endif
      // NOTE: the local mmap or CUDA allocation is controlled by the application
      free(private_buffer);
    }
  }

  // Disconnect
  socketClose(private_path->socket_fd);

  free(private_path);

  return success;
}

static bool postRecvRequest(TakyonPath *path, PrivateTakyonPath *private_path, TakyonRecvRequest *request) {
  // Total sub items in list
  uint32_t total_sub_items = path->attrs.max_pending_recv_requests * MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request);

  // Get structure shared by sender and receiver
  RecvRequestAndCompletion *local_request = &private_path->local_recv_request_and_completions[private_path->curr_local_unused_recv_request_index];

  // Fill in the sub buffer details
  local_request->sub_buffer_count = request->sub_buffer_count;
  for (uint32_t j=0; j<request->sub_buffer_count; j++) {
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[j];
    RecvRequestAndCompletion *local_sub_buffer = &private_path->local_recv_request_and_completions[private_path->curr_local_unused_recv_request_index + j];
#ifdef EXTRA_ERROR_CHECKING
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
#endif
    local_sub_buffer->buffer_index = sub_buffer->buffer_index;
    local_sub_buffer->bytes = sub_buffer->bytes;
    local_sub_buffer->offset = sub_buffer->offset;
  }

  // Make sure the remote side knows the transfer is not yet completed
  local_request->transfer_complete = false;

  // This will let the remote side know that the recv requestis posted
  local_request->transfer_posted = true;

  // Prepare for the next buffer
  private_path->curr_local_unused_recv_request_index = (private_path->curr_local_unused_recv_request_index + MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request)) % total_sub_items;

  return true;
}

bool interProcessCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  bool timed_out = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the provider
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(path->attrs.provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get provider name: %s\n", error_message);
    return false;
  }

  // See if unreliable
  bool is_unreliable = (strcmp(provider_name, "InterProcessUC") == 0);

  // -pathID=<non_negative_integer>
  uint32_t path_id = 0;
  bool path_id_found = false;
  bool ok = argGetUInt(path->attrs.provider, "-pathID=", &path_id, &path_id_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -pathID=<non_negative_integer> is invalid: %s\n", error_message);
    return false;
  }
  if (!path_id_found) {
    TAKYON_RECORD_ERROR(path->error_message, "Must specify -pathID=<non_negative_integer>\n");
    return false;
  }

  // Make sure enough room to post the initial recvs
  if (post_recv_count > path->attrs.max_pending_recv_requests) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough room to post the initial recv requests\n");
    return false;
  }

#ifdef ENABLE_CUDA
  if (MAX_CUDA_EVENTS < path->attrs.max_pending_recv_requests) {
    TAKYON_RECORD_ERROR(path->error_message, "MAX_CUDA_EVENTS < path->attrs.max_pending_recv_requests. May be better to increase MAX_CUDA_EVENTS in %s, versus reducing path->attrs.max_pending_recv_requests\n", __FILE__);
    return false;
  }
#endif

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;
  private_path->is_unreliable = is_unreliable;
  private_path->connection_failed = false;
  private_path->socket_fd = -1;
  private_path->remote_buffer_count = 0;
  private_path->remote_buffers = NULL;
  private_path->local_recv_request_and_completions = NULL;
  private_path->max_remote_pending_recv_requests = 0;
  private_path->max_remote_sub_buffers_per_recv_request = 0;
  private_path->remote_recv_request_and_completions = NULL;
  private_path->curr_local_unused_recv_request_index = 0;
  private_path->curr_local_posted_recv_request_index = 0;
  private_path->curr_remote_unused_recv_request_index = 0;
  private_path->curr_remote_posted_recv_request_index = 0;

  RemoteTakyonBufferInfo *remote_buffer_info_list = NULL;
  RemoteTakyonBufferInfo *local_buffer_info_list = NULL;

  // Make sure each buffer knows it's for this path: need for verifications later on
  // And if CUDA memory, then provide some extra CUDA info and resourses
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    buffer->private_data = NULL;
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    PrivateTakyonBuffer *private_buffer = calloc(1, sizeof(PrivateTakyonBuffer));
    if (private_buffer == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto cleanup;
    }
    buffer->private_data = private_buffer;
    private_buffer->path = path;
    bool is_cuda = false;
#ifdef ENABLE_CUDA
    if (!isCudaAddress(buffer->addr, &private_buffer->is_cuda, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "isCudaAddress() failed: %s\n", error_message);
      goto cleanup;
    }
    if (private_buffer->is_cuda) {
      is_cuda = true;
      for (uint32_t j=0; j<MAX_CUDA_EVENTS; j++) {
        if (!cudaEventAlloc(&private_buffer->cuda_event[j], error_message, MAX_ERROR_MESSAGE_CHARS)) {
          TAKYON_RECORD_ERROR(path->error_message, "cudaEventAlloc() failed: %s\n", error_message);
          goto cleanup;
        }
        private_buffer->cuda_event_allocated[j] = true;
      }
    }
#endif
    if (!is_cuda) {
      // Make sure buffer name is not empty
      if (strlen(buffer->name) == 0 || strlen(buffer->name) >= TAKYON_MAX_BUFFER_NAME_CHARS) {
        TAKYON_RECORD_ERROR(path->error_message, "attrs->buffers[%d].name is invalid\n", i);
        goto cleanup;
      }
    }
  }

  // Create the socket and connect with remote endpoint
  char local_socket_name[TAKYON_MAX_PROVIDER_CHARS];
  snprintf(local_socket_name, TAKYON_MAX_PROVIDER_CHARS, "InterProcessSocket_%d", path_id);
  if (path->attrs.is_endpointA) {
    if (!socketCreateLocalClient(local_socket_name, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create local client socket, needed to organize the InterProcess communication: %s\n", error_message);
      goto cleanup;
    }
  } else {
    if (!socketCreateLocalServer(local_socket_name, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create local server socket, needed to organize the InterProcess communication: %s\n", error_message);
      goto cleanup;
    }
  }

  // Prepare buffer information
  size_t local_buffer_info_list_bytes = path->attrs.buffer_count * sizeof(RemoteTakyonBufferInfo);
  local_buffer_info_list = malloc(local_buffer_info_list_bytes);
  if (local_buffer_info_list == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    goto cleanup;
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    RemoteTakyonBufferInfo *buffer_info = &local_buffer_info_list[i];
    buffer_info->bytes = buffer->bytes;
    memcpy(buffer_info->name, buffer->name, TAKYON_MAX_BUFFER_NAME_CHARS);
#ifdef ENABLE_CUDA
    PrivateTakyonBuffer *private_buffer = (PrivateTakyonBuffer *)buffer->private_data;
    if (private_buffer->is_cuda) {
      buffer_info->is_cuda = true;
      if (!cudaCreateIpcMapFromLocalAddr(buffer->addr, &buffer_info->ipc_addr_map, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "cudaCreateIpcMapFromLocalAddr() failed: %s\n", error_message);
        goto cleanup;
      }
      for (uint32_t j=0; j<MAX_CUDA_EVENTS; j++) {
        if (!cudaCreateIpcMapFromLocalEvent(&private_buffer->cuda_event[j], &buffer_info->ipc_event_map[j], error_message, MAX_ERROR_MESSAGE_CHARS)) {
          TAKYON_RECORD_ERROR(path->error_message, "cudaCreateIpcMapFromLocalEvent() failed: %s\n", error_message);
          goto cleanup;
        }
      }
    }
#endif
  }

  // Swap remote buffer info
  if (path->attrs.is_endpointA) {
    // Send buffer count
    if (!socketSend(private_path->socket_fd, &path->attrs.buffer_count, sizeof(path->attrs.buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local buffer count: %s\n", error_message);
      goto cleanup;
    }
    // Send buffer info
    if (!socketSend(private_path->socket_fd, local_buffer_info_list, local_buffer_info_list_bytes, false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local buffer info: %s\n", error_message);
      goto cleanup;
    }
    // Recv remote buffer count
    if (!socketRecv(private_path->socket_fd, &private_path->remote_buffer_count, sizeof(private_path->remote_buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote buffer count: %s\n", error_message);
      goto cleanup;
    }
    size_t remote_buffer_info_bytes = private_path->remote_buffer_count * sizeof(RemoteTakyonBufferInfo);
    remote_buffer_info_list = malloc(remote_buffer_info_bytes);
    if (remote_buffer_info_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto cleanup;
    }
    // Recv remote buffer info
    if (!socketRecv(private_path->socket_fd, remote_buffer_info_list, remote_buffer_info_bytes, false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote buffer info: %s\n", error_message);
      goto cleanup;
    }
  } else {
    // Recv remote buffer count
    if (!socketRecv(private_path->socket_fd, &private_path->remote_buffer_count, sizeof(private_path->remote_buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote buffer count: %s\n", error_message);
      goto cleanup;
    }
    size_t remote_buffer_info_bytes = private_path->remote_buffer_count * sizeof(RemoteTakyonBufferInfo);
    remote_buffer_info_list = malloc(remote_buffer_info_bytes);
    if (remote_buffer_info_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto cleanup;
    }
    // Recv remote buffer info
    if (!socketRecv(private_path->socket_fd, remote_buffer_info_list, remote_buffer_info_bytes, false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote buffer info: %s\n", error_message);
      goto cleanup;
    }
    // Send buffer count
    if (!socketSend(private_path->socket_fd, &path->attrs.buffer_count, sizeof(path->attrs.buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local buffer count: %s\n", error_message);
      goto cleanup;
    }
    // Send buffer info
    if (!socketSend(private_path->socket_fd, local_buffer_info_list, local_buffer_info_list_bytes, false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local buffer info: %s\n", error_message);
      goto cleanup;
    }
  }
  // No need for the local list anymore
  free(local_buffer_info_list);
  local_buffer_info_list = NULL;

  // Get memory mapped addresses of remote buffers
  private_path->remote_buffers = calloc(private_path->remote_buffer_count, sizeof(RemoteTakyonBuffer));
  if (private_path->remote_buffers == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    goto cleanup;
  }
  for (uint32_t i=0; i<private_path->remote_buffer_count; i++) {
    RemoteTakyonBufferInfo *remote_buffer_info = &remote_buffer_info_list[i];
    RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[i];
    remote_buffer->bytes = remote_buffer_info->bytes;
    bool is_cuda = false;
#ifdef ENABLE_CUDA
    remote_buffer->is_cuda = remote_buffer_info->is_cuda;
    is_cuda = remote_buffer->is_cuda;
    if (is_cuda) {
      remote_buffer->mmap_addr = cudaGetRemoteAddrFromIpcMap(remote_buffer_info->ipc_addr_map, error_message, MAX_ERROR_MESSAGE_CHARS);
      if (remote_buffer->mmap_addr == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "cudaGetRemoteAddrFromIpcMap() failed: %s\n", error_message);
        goto cleanup;
      }
      for (uint32_t j=0; j<MAX_CUDA_EVENTS; j++) {
        if (!cudaGetRemoteEventFromIpcMap(remote_buffer_info->ipc_event_map[j], &remote_buffer->cuda_event[j], error_message, MAX_ERROR_MESSAGE_CHARS)) {
          TAKYON_RECORD_ERROR(path->error_message, "cudaGetRemoteEventFromIpcMap() failed: %s\n", error_message);
          goto cleanup;
        }
      }
    }
#endif
    if (!is_cuda) {
      bool got_it = false;
      if (!mmapGet(remote_buffer_info->name, remote_buffer->bytes, &remote_buffer->mmap_addr, &got_it, &remote_buffer->mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "mmapGet() failed: %s\n", error_message);
        goto cleanup;
      }
      if (!got_it) {
        TAKYON_RECORD_ERROR(path->error_message, "Memory map '%s' does not exist. Make sure the app uses mmapAlloc() from src/utils/utils_ipc.h, to create the TakyonBuffer instead of malloc()\n", remote_buffer_info->name);
        goto cleanup;
      }
    }
  }
  // No need for the list any more
  free(remote_buffer_info_list);
  remote_buffer_info_list = NULL;

  // Allocate mmap for tracking posted recvs
  RemotePathInfo local_path_info;
  memset(&local_path_info, 0, sizeof(RemotePathInfo));
  if (path->attrs.max_pending_recv_requests > 0) {
    local_path_info.max_pending_recv_requests = path->attrs.max_pending_recv_requests;
    local_path_info.max_sub_buffers_per_recv_request = path->attrs.max_sub_buffers_per_recv_request;
    uint32_t local_request_element_count = path->attrs.max_pending_recv_requests * MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request);
    uint64_t local_request_list_bytes = local_request_element_count * sizeof(RecvRequestAndCompletion);
    snprintf(local_path_info.mmap_name, TAKYON_MAX_BUFFER_NAME_CHARS, "TakyonPath_%s_%u_" UINT64_FORMAT, path->attrs.is_endpointA ? "A" : "B", path_id, local_request_list_bytes); // IMPORTANT: Must be unique to all named mmaps in OS
    if (!mmapAlloc(local_path_info.mmap_name, local_request_list_bytes, (void **)&private_path->local_recv_request_and_completions, &private_path->local_postings_mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "mmapAlloc() failed: %s\n", error_message);
      goto cleanup;
    }

    // Init post recv list before it gets used
    for (uint32_t i=0; i<local_request_element_count; i++) {
      RecvRequestAndCompletion *element = &private_path->local_recv_request_and_completions[i];
      element->transfer_posted = false;
    }

    // Post the inital recvs
    for (uint32_t i=0; i<post_recv_count; i++) {
      TakyonRecvRequest *request = &recv_requests[i];
      if (!postRecvRequest(path, private_path, request)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to post initial recv requests\n");
        goto cleanup;
      }
    }
  }

  // Swap remote path info
  RemotePathInfo remote_path_info;
  if (path->attrs.is_endpointA) {
    // Send
    if (!socketSend(private_path->socket_fd, &local_path_info, sizeof(RemotePathInfo), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local path info: %s\n", error_message);
      goto cleanup;
    }
    // Recv
    if (!socketRecv(private_path->socket_fd, &remote_path_info, sizeof(RemotePathInfo), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote path info: %s\n", error_message);
      goto cleanup;
    }
  } else {
    // Recv
    if (!socketRecv(private_path->socket_fd, &remote_path_info, sizeof(RemotePathInfo), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote path info: %s\n", error_message);
      goto cleanup;
    }
    // Send
    if (!socketSend(private_path->socket_fd, &local_path_info, sizeof(RemotePathInfo), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send local path info: %s\n", error_message);
      goto cleanup;
    }
  }

  // Get the remote mmap to the remote path info
  private_path->max_remote_pending_recv_requests = remote_path_info.max_pending_recv_requests;
  private_path->max_remote_sub_buffers_per_recv_request = remote_path_info.max_sub_buffers_per_recv_request;
  if (private_path->max_remote_pending_recv_requests > 0) {
    uint64_t remote_request_list_bytes = private_path->max_remote_pending_recv_requests * MY_MAX(1, private_path->max_remote_sub_buffers_per_recv_request) * sizeof(RecvRequestAndCompletion);
    bool got_it = false;
    if (!mmapGet(remote_path_info.mmap_name, remote_request_list_bytes, (void **)&private_path->remote_recv_request_and_completions, &got_it, &private_path->remote_postings_mmap_handle, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "mmapGet() failed: %s\n", error_message);
      goto cleanup;
    }
    if (!got_it) {
      TAKYON_RECORD_ERROR(path->error_message, "Memory map '%s' does not exist\n", remote_path_info.mmap_name);
      goto cleanup;
    }
  }

  // Start the thread to detect if the socket is disconnected
  int rc = pthread_create(&private_path->disconnect_detection_thread_id, NULL, disconnectDetectionThread, private_path);
  if (rc != 0) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start disconnectDetectionThread(): rc=%d\n", rc);
    goto cleanup;
  }
  private_path->thread_started = true;

  // Ready to start transferring
  return true;

 cleanup:
  // Error occured
  (void)freeResources(path, private_path, remote_buffer_info_list, local_buffer_info_list, false);
  return false;
}

bool interProcessDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  bool timed_out = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Wake up thread
  if (private_path->thread_started) {
    if (!private_path->connection_failed) {
      // Wake thread up so it will exit
      uint32_t dummy = 0;
      if (!socketSend(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to wake up disconnectDetectionThread(): %s\n", error_message);
        private_path->connection_failed = true;
      }
    }
    // Wait for the thread to exit
    int rc = pthread_join(private_path->disconnect_detection_thread_id, NULL);
    if (rc != 0) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to join disconnectDetectionThread(): rc=%d\n", rc);
      private_path->connection_failed = true;
    }
  }

  // Socket barrier: to make sure pending transactions are complete
  bool barrier_ok = !private_path->connection_failed;
  if (barrier_ok) {
    if (path->attrs.is_endpointA) {
      uint32_t x = 23;
      // Send
      if (!socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barriar value: %s\n", error_message);
	return false;
      }
      x = 33;
      // Recv
      if (barrier_ok && !socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
	return false;
      }
      if (x != 23) {
        TAKYON_RECORD_ERROR(path->error_message, "Got incorrect barrier value: %s\n", error_message);
	return false;
      }
    } else {
      uint32_t x = 33;
      // Recv
      if (!socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
	return false;
      }
      // Send
      if (barrier_ok && !socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barrier value: %s\n", error_message);
	return false;
      }
    }
  } else {
    TAKYON_RECORD_ERROR(path->error_message, "This endpoint is trying to finalize, but the remote side seems to have disconnected\n");
    return false;
  }

  // Free the resources
  bool ok = freeResources(path, private_path, NULL, NULL, true);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to destroy path\n");
    return false;
  }

  return true;
}

static bool transferData(void *dest_addr, void *src_addr, uint64_t bytes, char *error_message) {
#ifdef ENABLE_CUDA
  // IMPORTANT: this will handle all combinations: CPU/GPU -> CPU/GPU; i.e. CPU -> CPU will work like memcpy()
  cudaError_t cuda_status = cudaMemcpy(dest_addr, src_addr, bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) {
    TAKYON_RECORD_ERROR(error_message, "Failed to transfer %jd bytes with cudaMemcpy(): %s\n", bytes, cudaGetErrorString(cuda_status));
    return false;
  }
#else
  if (memcpy(dest_addr, src_addr, bytes) != dest_addr) {
    TAKYON_RECORD_ERROR(error_message, "Failed to transfer %jd bytes with memcpy()\n", bytes);
    return false;
  }
#endif
  return true;
}

bool interProcessOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds;
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

#ifdef EXTRA_ERROR_CHECKING
  // Make sure at least one buffer
  if (request->sub_buffer_count == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "One sided requests must have at least one sub buffer\n");
    return false;
  }
#endif

  // Get total bytes to transfer
  uint64_t total_local_bytes_to_transfer = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    uint64_t local_bytes = sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *local_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    PrivateTakyonBuffer *local_private_buffer = (PrivateTakyonBuffer *)local_buffer->private_data;
    if (local_private_buffer->path != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
    if (local_bytes > (local_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds local buffer (bytes = %ju)\n", local_bytes, sub_buffer->offset, local_buffer->bytes);
      return false;
    }
#endif
    total_local_bytes_to_transfer += local_bytes;
  }

  // Remote info
#ifdef EXTRA_ERROR_CHECKING
  if (request->remote_buffer_index >= private_path->remote_buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "Remote buffer index = %d is out of range\n", request->remote_buffer_index);
    return false;
  }
#endif
  RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[request->remote_buffer_index];
  void *remote_addr = (void *)((uint64_t)remote_buffer->mmap_addr + request->remote_offset);
  uint64_t remote_max_bytes = remote_buffer->bytes - request->remote_offset;

  // Verify enough space in remote request
  if (total_local_bytes_to_transfer > remote_max_bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough available remote bytes\n");
    return false;
  }

  // Copy the data to the remote side
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    TakyonBuffer *local_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    void *local_addr = (void *)((uint64_t)local_buffer->addr + sub_buffer->offset);
    uint64_t bytes = sub_buffer->bytes;
    if (request->operation == TAKYON_OP_WRITE) {
      if (!transferData(remote_addr, local_addr, bytes, path->error_message)) return false;
    } else if (request->operation == TAKYON_OP_READ) {
      if (!transferData(local_addr, remote_addr, bytes, path->error_message)) return false;
    } else {
      /*+ add support for atomics */
      TAKYON_RECORD_ERROR(path->error_message, "One sided operation '%s' not supported\n", takyonPrivateOneSidedOpToText(request->operation));
      return false;
    }
    remote_addr = (void *)((uint64_t)remote_addr + bytes);
  }

  // NOTES:
  // - If pulling, and local memory is CUDA, it should be synchronous
  // - If pushing, and remote memory is CUDA, the remote side will need the app to use some CUDA synchronization to gaurantee it arrived

  return true;
}

bool interProcessSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds;
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Get total bytes to send
  uint64_t total_bytes_to_send = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    uint64_t src_bytes = sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    PrivateTakyonBuffer *src_private_buffer = (PrivateTakyonBuffer *)src_buffer->private_data;
    if (src_private_buffer->path != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
    if (src_bytes > (src_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds src buffer (bytes = %ju)\n", src_bytes, sub_buffer->offset, src_buffer->bytes);
      return false;
    }
#endif
    total_bytes_to_send += src_bytes;
  }

  // Get the current posted request
  uint32_t total_sub_items = private_path->max_remote_pending_recv_requests * MY_MAX(1, private_path->max_remote_sub_buffers_per_recv_request);
  RecvRequestAndCompletion *remote_request = &private_path->remote_recv_request_and_completions[private_path->curr_remote_posted_recv_request_index];
  if (!remote_request->transfer_posted || remote_request->transfer_complete) {
    if (private_path->is_unreliable) {
      return true;
    } else {
      TAKYON_RECORD_ERROR(path->error_message, "Remote side has no posted recvs\n");
      return false;
    }
  }

  // Get total available recv bytes
  uint64_t total_available_recv_bytes = 0;
  for (uint32_t i=0; i<remote_request->sub_buffer_count; i++) {
    RecvRequestAndCompletion *remote_sub_buffer = &private_path->remote_recv_request_and_completions[private_path->curr_remote_posted_recv_request_index + i];
    uint64_t remote_max_bytes = remote_sub_buffer->bytes;
    RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[remote_sub_buffer->buffer_index];
    if (remote_max_bytes > (remote_buffer->bytes - remote_sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds remote buffer (bytes = %ju)\n", remote_max_bytes, remote_sub_buffer->offset, remote_buffer->bytes);
      return false;
    }
#ifdef ENABLE_CUDA
    if (remote_buffer->is_cuda) {
      remote_buffer->send_notification = false; // Prepare for this memory to be used
    }
#endif
    total_available_recv_bytes += remote_max_bytes;
  }

  // Verify enough space in remote request
  if (total_bytes_to_send > total_available_recv_bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough available bytes in remote request\n");
    return false;
  }

  // Get a handle to the first remote memory block
  uint32_t remote_sub_buffer_index = private_path->curr_remote_posted_recv_request_index;
  RecvRequestAndCompletion *remote_sub_buffer = &private_path->remote_recv_request_and_completions[remote_sub_buffer_index];
  RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[remote_sub_buffer->buffer_index];
  void *remote_addr = (void *)((uint64_t)remote_buffer->mmap_addr + remote_sub_buffer->offset);
  uint64_t remote_max_bytes = remote_sub_buffer->bytes;

  // Copy the data to the remote side
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    void *src_addr = (void *)((uint64_t)src_buffer->addr + sub_buffer->offset);
    uint64_t src_bytes = sub_buffer->bytes;
    while (src_bytes > 0) {
      if (remote_max_bytes == 0) {
        remote_sub_buffer_index++;
        remote_sub_buffer = &private_path->remote_recv_request_and_completions[remote_sub_buffer_index];
        remote_buffer = &private_path->remote_buffers[remote_sub_buffer->buffer_index];
        remote_addr = (void *)((uint64_t)remote_buffer->mmap_addr + remote_sub_buffer->offset);
        remote_max_bytes = remote_sub_buffer->bytes;
      }
      uint64_t bytes_to_send = src_bytes;
      if (remote_max_bytes < bytes_to_send) bytes_to_send = remote_max_bytes;
      if (!transferData(remote_addr, src_addr, bytes_to_send, path->error_message)) {
        return false;
      }
#ifdef ENABLE_CUDA
      if (remote_buffer->is_cuda) {
        remote_buffer->send_notification = true; // This cuda buffer was used, so will need to send notificastion after the transfer has been started
      }
#endif
      src_addr = (void *)((uint64_t)src_addr + bytes_to_send);
      src_bytes -= bytes_to_send;
      remote_addr = (void *)((uint64_t)remote_addr + bytes_to_send);
      remote_max_bytes -= bytes_to_send;
    }
  }

  // See if any CUDA event notifications need to be sent (if the remote CUDA buffers had data transfered to them)
#ifdef ENABLE_CUDA
  {
    char error_message[MAX_ERROR_MESSAGE_CHARS];
    for (uint32_t i=0; i<remote_request->sub_buffer_count; i++) {
      RecvRequestAndCompletion *remote_sub_buffer = &private_path->remote_recv_request_and_completions[private_path->curr_remote_posted_recv_request_index + i];
      RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[remote_sub_buffer->buffer_index];
      if (remote_buffer->is_cuda && remote_buffer->send_notification) {
        // Data was sent to this remote CUDA buffer
        uint32_t event_index = remote_buffer->curr_cuda_event_index;
        // Verify the event is still not be used for a previous transfer
        if (!cudaEventAvailable(&remote_buffer->cuda_event[event_index], error_message, MAX_ERROR_MESSAGE_CHARS)) {
          TAKYON_RECORD_ERROR(path->error_message, "cudaEventNotify() failed: %s\n", error_message);
          return false;
        }
        // Activate the event
        if (!cudaEventNotify(&remote_buffer->cuda_event[event_index], error_message, MAX_ERROR_MESSAGE_CHARS)) {
          TAKYON_RECORD_ERROR(path->error_message, "cudaEventNotify() failed: %s\n", error_message);
          return false;
        }
        remote_buffer->curr_cuda_event_index = (remote_buffer->curr_cuda_event_index + 1) % MAX_CUDA_EVENTS;
        remote_buffer->send_notification = false;
      }
    }
  }
#endif

  // Set the request results
  remote_request->bytes_received = total_bytes_to_send;
  remote_request->piggyback_message = piggyback_message;

  // Mark the request as complete
  remote_request->transfer_complete = true;

  // Prepare for the next request
  private_path->curr_remote_posted_recv_request_index = (private_path->curr_remote_posted_recv_request_index + MY_MAX(1, private_path->max_remote_sub_buffers_per_recv_request)) % total_sub_items;

  return true;
}

bool interProcessPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Total sub items in list
  uint32_t total_sub_items = path->attrs.max_pending_recv_requests * MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request);

  // Verify enough room to post
  uint32_t local_index = private_path->curr_local_unused_recv_request_index;
  uint32_t temp_request_count = request_count;
  while (temp_request_count > 0) {
    RecvRequestAndCompletion *local_request = &private_path->local_recv_request_and_completions[local_index];
    if (local_request->transfer_posted) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of room to post recv requests (attrs.max_pending_recv_requests = %d). May need to increase attrs.max_pending_recv_requests\n", path->attrs.max_pending_recv_requests);
      return false;
    }
    temp_request_count--;
    local_index = (local_index + MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request)) % total_sub_items;
  }

  // Put the requests on the list
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *request = &requests[i];
    if (!postRecvRequest(path, private_path, request)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to post recv request\n");
      return false;
    }
  }

  return true;
}

bool interProcessIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret) {
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  int64_t time1 = clockTimeNanoseconds();

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Get the current posted local request
  uint32_t total_sub_items = path->attrs.max_pending_recv_requests * MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request);
  RecvRequestAndCompletion *local_request = &private_path->local_recv_request_and_completions[private_path->curr_local_posted_recv_request_index];

  // See if the data has been sent
  while (!local_request->transfer_complete && !private_path->connection_failed) {
    // No data yet, so wait for data until the timeout occurs
    // IMPORTANT: current need to use polling even if even driven. Just make the sleep amount reasonable if event driven
    // NOT USED: if (request->use_polling_completion) {
    // Check timeout
    if (timeout_nano_seconds == 0) {
      // No timeout, so return now
      *timed_out_ret = true;
      return true;
    } else if (timeout_nano_seconds >= 0) {
      // Hit the timeout without data, time to return
      int64_t time2 = clockTimeNanoseconds();
      int64_t diff = time2 - time1;
      if (diff > timeout_nano_seconds) {
        *timed_out_ret = true;
        return true;
      }
    }
    // Data not ready. In polling mode, so sleep a little to avoid buring up CPU core
    if (request->use_polling_completion) {
      if (request->usec_sleep_between_poll_attempts > 0) clockSleepUsecs(request->usec_sleep_between_poll_attempts);
    } else {
      // KLUDGE: This is event driven so don't want polling to be intrusive, but still need to be pretty resposive
      clockSleepUsecs(KLUDGE_USEC_SLEEP_BETWEEN_POLL_ATTEMPTS);
    }
  }

  // Data was sent. CPU transfers are synchronous, but CUDA transfers are asynchronous

  // For each CUDA buffer, need to wait on a sync event, but only for bytes received (i.e. more recv bytes may have been posted than were received, so not all sub buffers may get data
#ifdef ENABLE_CUDA
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  uint64_t total_available_recv_bytes = 0;
  for (uint32_t i=0; i<local_request->sub_buffer_count; i++) {
    RecvRequestAndCompletion *local_sub_buffer = &private_path->local_recv_request_and_completions[private_path->curr_local_posted_recv_request_index + i];
    uint64_t local_max_bytes = local_sub_buffer->bytes;
    TakyonBuffer *local_buffer = &path->attrs.buffers[local_sub_buffer->buffer_index];
    PrivateTakyonBuffer *private_buffer = (PrivateTakyonBuffer *)local_buffer->private_data;
    if (private_buffer->is_cuda && local_request->bytes_received > total_available_recv_bytes) {
      // Data was copied to this CUDA buffer, so need to know that it arrived
      uint32_t event_index = private_buffer->curr_cuda_event_index;
      if (!cudaEventWait(&private_buffer->cuda_event[event_index], error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "cudaEventWait() failed: %s\n", error_message);
        return false;
      }
      private_buffer->curr_cuda_event_index = (private_buffer->curr_cuda_event_index + 1) % MAX_CUDA_EVENTS;
    }
    total_available_recv_bytes += local_max_bytes;
  }
#endif

  // Prepare for the next request
  private_path->curr_local_posted_recv_request_index = (private_path->curr_local_posted_recv_request_index + MY_MAX(1, path->attrs.max_sub_buffers_per_recv_request)) % total_sub_items;

  // Return results
  *bytes_received_ret = local_request->bytes_received;
  *piggyback_message_ret = local_request->piggyback_message;

  // Mark the request as unused
  local_request->transfer_posted = false;

  return true;
}
