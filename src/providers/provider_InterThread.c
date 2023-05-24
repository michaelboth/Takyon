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

#include "provider_InterThread.h"
#include "utils_inter_thread_manager.h"
#include "takyon_private.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include "utils_thread_cond_timed_wait.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef ENABLE_CUDA
  #include "cuda_runtime.h"
#endif

// attrs->provider[] Specification:
//   "InterThreadRC -pathID=<non_negative_integer>"
//   "InterThreadUC -pathID=<non_negative_integer>"

#define THREAD_MANAGER_ID_RELIABLE   23 // This must be different from the other providers that use the thread manager
#define THREAD_MANAGER_ID_UNRELIABLE 24 // This must be different from the other providers that use the thread manager

typedef struct {
  bool transfer_complete;
  uint64_t bytes_received;
  uint32_t piggyback_message;
} RecvCompletion;

typedef struct {
  InterThreadManagerItem *remote_thread_handle;
  bool is_unreliable;
  uint32_t posted_recv_count;
  uint32_t oldest_posted_recv_index;
  uint32_t newest_posted_recv_index;
  TakyonRecvRequest **posted_recvs; // Circular buffer
  RecvCompletion *recv_completions; // Circular buffer lock step with posted_recvs
} PrivateTakyonPath;

bool interThreadCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the provider
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(path->attrs.provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get provider name: %s\n", error_message);
    return false;
  }

  // See if unreliable
  bool is_unreliable = (strcmp(provider_name, "InterThreadUC") == 0);

  // Get the path ID
  uint32_t path_id;
  bool found;
  bool ok = argGetUInt(path->attrs.provider, "-pathID=", &path_id, &found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -pathID=<non_negative_integer> is invalid: %s\n", error_message);
    return false;
  }
  if (!found) {
    TAKYON_RECORD_ERROR(path->error_message, "Must specify -pathID=<non_negative_integer>\n");
    return false;
  }

  // Make sure each buffer knows it's for this path: need for verifications later on
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    path->attrs.buffers[i].private_data = path;
  }

  // Make sure enough room to post the initial recvs
  if (post_recv_count > path->attrs.max_pending_recv_requests) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough room to post the initial recv requests\n");
    return false;
  }

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;
  private_path->is_unreliable = is_unreliable;
  private_path->posted_recv_count = 0;
  private_path->oldest_posted_recv_index = 0;
  private_path->newest_posted_recv_index = 0;

  // Create the posted_recvs and recv completions
  if (path->attrs.max_pending_recv_requests > 0) {
    private_path->posted_recvs = (TakyonRecvRequest**)malloc(path->attrs.max_pending_recv_requests * sizeof(TakyonRecvRequest*));
    if (private_path->posted_recvs == NULL) {
      free(private_path);
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      return false;
    }
    private_path->recv_completions = (RecvCompletion*)malloc(path->attrs.max_pending_recv_requests * sizeof(RecvCompletion));
    if (private_path->recv_completions == NULL) {
      free(private_path->posted_recvs);
      free(private_path);
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      return false;
    }
  }

  // Post the recvs before making the connection
  for (uint32_t i=0; i<post_recv_count; i++) {
    TakyonRecvRequest *request = &recv_requests[i];
    private_path->posted_recvs[private_path->newest_posted_recv_index] = request;
    private_path->recv_completions[private_path->newest_posted_recv_index].transfer_complete = false;
    request->private_data = (void *)((uint64_t)private_path->newest_posted_recv_index);
    private_path->newest_posted_recv_index = (private_path->newest_posted_recv_index + 1) % path->attrs.max_pending_recv_requests; // Move to the next item
    private_path->posted_recv_count++;
  }

  // Call this to make sure the mutex manager is ready to coordinate: This can be called multiple times, but it's guaranteed to atomically run only the first time called.
  if (!interThreadManagerInit()) {
    if (private_path->recv_completions != NULL) free(private_path->recv_completions);
    if (private_path->posted_recvs != NULL) free(private_path->posted_recvs);
    free(private_path);
    TAKYON_RECORD_ERROR(path->error_message, "failed to start the inter-thread mutex manager\n");
    return false;
  }

  // Connect to the remote thread
  uint32_t provider_id = is_unreliable ? THREAD_MANAGER_ID_UNRELIABLE : THREAD_MANAGER_ID_RELIABLE;
  private_path->remote_thread_handle = interThreadManagerConnect(provider_id, path_id, path, timeout_seconds);
  if (private_path->remote_thread_handle == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to connect to remote thread\n");
    goto cleanup;
  }

  // Ready to start transferring
  return true;

 cleanup:
  // An error ocurred so clean up all allocated resources

  // Let the thread manager know it's done with this thread
  interThreadManagerFinalize();
  if (private_path->recv_completions != NULL) free(private_path->recv_completions);
  if (private_path->posted_recvs != NULL) free(private_path->posted_recvs);
  free(private_path);

  return false;
}

bool interThreadDestroy(TakyonPath *path, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // Do a coordinated disconnect
  bool graceful_disconnect_ok = true;
  if (!interThreadManagerDisconnect(path, private_path->remote_thread_handle, timeout_seconds)) {
    TAKYON_RECORD_ERROR(path->error_message, "Thread disconnect failed\n");
    graceful_disconnect_ok = false;
  }

  // Let the thread manager know it's done with this thread
  interThreadManagerFinalize();
  if (private_path->recv_completions != NULL) free(private_path->recv_completions);
  if (private_path->posted_recvs != NULL) free(private_path->posted_recvs);
  free(private_path);

  return graceful_disconnect_ok;
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

static bool doTwoSidedTransfer(TakyonPath *path, TakyonPath *remote_path, TakyonSendRequest *request, uint32_t piggyback_message, InterThreadManagerItem *remote_thread_handle) {
  // IMPORTANT: mutex is locked at this point

  // Keep track of the remote memory blocks
  TakyonComm *remote_comm = (TakyonComm *)remote_path->private_data;
  PrivateTakyonPath *remote_private_path = (PrivateTakyonPath *)remote_comm->data;
  if (remote_private_path->posted_recv_count == 0) {
    if (remote_private_path->is_unreliable) {
      return true;
    } else {
      TAKYON_RECORD_ERROR(path->error_message, "Remote side has no posted recvs\n");
      return false;
    }
  }
  // Pull the request off the list
  TakyonRecvRequest *remote_request = remote_private_path->posted_recvs[remote_private_path->oldest_posted_recv_index];
  remote_private_path->oldest_posted_recv_index = (remote_private_path->oldest_posted_recv_index + 1) % remote_path->attrs.max_pending_recv_requests; // Move to next item in list
  remote_private_path->posted_recv_count--;

  // Get total bytes to send
  uint64_t total_bytes_to_send = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    uint64_t src_bytes = sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    if (src_buffer->private_data != path) {
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

  // Get total available recv bytes
  uint64_t total_available_recv_bytes = 0;
  for (uint32_t i=0; i<remote_request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *remote_sub_buffer = &remote_request->sub_buffers[i];
    uint64_t remote_max_bytes = remote_sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (remote_sub_buffer->buffer_index >= remote_path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'remote_sub_buffer->buffer_index == %d out of range\n", remote_sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *remote_buffer = &remote_path->attrs.buffers[remote_sub_buffer->buffer_index];
    if (remote_buffer->private_data != remote_path) {
      TAKYON_RECORD_ERROR(path->error_message, "'remote sub_buffers[%d] is not from the remote Takyon path\n", i);
      return false;
    }
    if (remote_max_bytes > (remote_buffer->bytes - remote_sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds remote buffer (bytes = %ju)\n", remote_max_bytes, remote_sub_buffer->offset, remote_buffer->bytes);
      return false;
    }
#endif
    total_available_recv_bytes += remote_max_bytes;
  }

  // Verify enough space in remote request
  if (total_bytes_to_send > total_available_recv_bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough available bytes in remote request\n");
    return false;
  }

  if (request->sub_buffer_count > 0) {
    // Get a handle to the first remote memory block
    uint32_t remote_sub_buffer_index = 0;
    TakyonSubBuffer *remote_sub_buffer = &remote_request->sub_buffers[remote_sub_buffer_index];
    TakyonBuffer *remote_buffer = &remote_path->attrs.buffers[remote_sub_buffer->buffer_index];
    void *remote_addr = (void *)((uint64_t)remote_buffer->addr + remote_sub_buffer->offset);
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
          remote_sub_buffer = &remote_request->sub_buffers[remote_sub_buffer_index];
          remote_buffer = &remote_path->attrs.buffers[remote_sub_buffer->buffer_index];
          remote_addr = (void *)((uint64_t)remote_buffer->addr + remote_sub_buffer->offset);
          remote_max_bytes = remote_sub_buffer->bytes;
        }
        uint64_t bytes_to_send = src_bytes;
        if (remote_max_bytes < bytes_to_send) bytes_to_send = remote_max_bytes;
        if (!transferData(remote_addr, src_addr, bytes_to_send, path->error_message)) {
          return false;
        }
        src_addr = (void *)((uint64_t)src_addr + bytes_to_send);
        src_bytes -= bytes_to_send;
        remote_addr = (void *)((uint64_t)remote_addr + bytes_to_send);
        remote_max_bytes -= bytes_to_send;
      }
    }
  }

  // Set the request results
  uint64_t remote_post_index = (uint64_t)remote_request->private_data;
  remote_private_path->recv_completions[remote_post_index].bytes_received = total_bytes_to_send;
  remote_private_path->recv_completions[remote_post_index].piggyback_message = piggyback_message;

  // Mark the request as complete
  remote_private_path->recv_completions[remote_post_index].transfer_complete = true;
  if (!remote_request->use_polling_completion) {
    // Signal receiver
    pthread_cond_signal(&remote_thread_handle->cond);
  }

  return true;
}

static bool doOneSidedTransfer(TakyonPath *path, TakyonPath *remote_path, TakyonOneSidedRequest *request) {
  // IMPORTANT: mutex is locked at this point

  // Make sure at least one buffer
  if (request->sub_buffer_count == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "One sided requests must have at least one sub buffer\n");
    return false;
  }

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
    if (local_buffer->private_data != path) {
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
  if (request->remote_buffer_index >= remote_path->attrs.buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "Remote buffer index = %d is out of range\n", request->remote_buffer_index);
    return false;
  }
#endif
  TakyonBuffer *remote_buffer = &remote_path->attrs.buffers[request->remote_buffer_index];
#ifdef EXTRA_ERROR_CHECKING
  if (remote_buffer->private_data != remote_path) {
    TAKYON_RECORD_ERROR(path->error_message, "Remote buffer is for a different Takyon path\n");
    return false;
  }
#endif
  void *remote_addr = (void *)((uint64_t)remote_buffer->addr + request->remote_offset);
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
      /*+ add support for atomics and write with piggyback */
      TAKYON_RECORD_ERROR(path->error_message, "One sided operation '%s' not supported\n", takyonPrivateOneSidedOpToText(request->operation));
      return false;
    }
    remote_addr = (void *)((uint64_t)remote_addr + bytes);
  }

  return true;
}

bool interThreadOneSided(TakyonPath *path, TakyonOneSidedRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds;   // Quiet compiler
  (void)piggyback_message; // Quiet compiler
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  InterThreadManagerItem *remote_thread_handle = private_path->remote_thread_handle;

  // Lock the mutex now since many of the variables come from the remote side
  pthread_mutex_lock(&remote_thread_handle->mutex); // IMPORTANT: this is held for a long time and will slow down takyonPostRecvs() takyonIsRecved()

  // Verify connection is good
  if (remote_thread_handle->connection_broken) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Remote side has failed\n");
    return false;
  }

  // Process request
  TakyonPath *remote_path = path->attrs.is_endpointA ? remote_thread_handle->pathB : remote_thread_handle->pathA;
  if (!doOneSidedTransfer(path, remote_path, request)) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "One sided transfer failed\n");
    return false;
  }

  // Done with remote variables
  pthread_mutex_unlock(&remote_thread_handle->mutex);

  return true;
}

bool interThreadSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds;
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  InterThreadManagerItem *remote_thread_handle = private_path->remote_thread_handle;

  // Lock the mutex now since many of the variables come from the remote side
  pthread_mutex_lock(&remote_thread_handle->mutex); // IMPORTANT: this is held for a long time and will slow down takyonPostRecvs() takyonIsRecved()

  // Verify connection is good
  if (remote_thread_handle->connection_broken) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Remote side has failed\n");
    return false;
  }

  // Do the transfer
  TakyonPath *remote_path = path->attrs.is_endpointA ? remote_thread_handle->pathB : remote_thread_handle->pathA;
  if (!doTwoSidedTransfer(path, remote_path, request, piggyback_message, remote_thread_handle)) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Send failed\n");
    return false;
  }

  // Done with remote variables
  pthread_mutex_unlock(&remote_thread_handle->mutex);

  return true;
}

bool interThreadPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  InterThreadManagerItem *remote_thread_handle = private_path->remote_thread_handle;

  // Lock the mutex now since many of the variables come from the remote side
  pthread_mutex_lock(&remote_thread_handle->mutex);

  // Verify connection is good
  if (remote_thread_handle->connection_broken) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Remote side has failed\n");
    return false;
  }

  // Verify enough room to post
  uint32_t remaining_count = (path->attrs.max_pending_recv_requests - private_path->posted_recv_count);
  if (request_count > remaining_count) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Out of room to post recv requests. May need to increase attrs.max_pending_recv_requests\n");
    return false;
  }

  // Put the requests on the list
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *request = &requests[i];
    private_path->posted_recvs[private_path->newest_posted_recv_index] = request;
    private_path->recv_completions[private_path->newest_posted_recv_index].transfer_complete = false;
    request->private_data = (void *)((uint64_t)private_path->newest_posted_recv_index);
    private_path->newest_posted_recv_index = (private_path->newest_posted_recv_index + 1) % path->attrs.max_pending_recv_requests; // Go to the next item
    private_path->posted_recv_count++;
  }

  // Done with remote variables
  pthread_mutex_unlock(&remote_thread_handle->mutex);

  return true;
}

bool interThreadIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret) {
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  InterThreadManagerItem *remote_thread_handle = private_path->remote_thread_handle;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  int64_t time1 = clockTimeNanoseconds();
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Lock the mutex now since many of the variables come from the remote side
  if (!request->use_polling_completion) {
    pthread_mutex_lock(&remote_thread_handle->mutex);
  }

  // IMPORTANT: If polling, the mutex is unlocked while spinning on 'waiting for data' or 'the connection is broken',
  //            but should be fine since both are single integers and mutually exclusive
  // See if the data has been sent
  uint64_t post_index = (uint64_t)request->private_data;
  while (!private_path->recv_completions[post_index].transfer_complete && !remote_thread_handle->connection_broken) {
    // No data yet, so wait for data until the timeout occurs
    if (request->use_polling_completion) {
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
      if (request->usec_sleep_between_poll_attempts > 0) clockSleepUsecs(request->usec_sleep_between_poll_attempts);
    } else {
      // Sleep while waiting for data
      bool timed_out = false;
      bool suceeded = threadCondWait(&remote_thread_handle->mutex, &remote_thread_handle->cond, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS);
      if (!suceeded) {
        interThreadManagerMarkConnectionAsBad(remote_thread_handle);
        pthread_mutex_unlock(&remote_thread_handle->mutex);
        TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for data: %s\n", error_message);
        return false;
      }
      if (timed_out) {
        pthread_mutex_unlock(&remote_thread_handle->mutex);
        *timed_out_ret = true;
        return true;
      }
    }
  }

  // Verify connection is good
  if (remote_thread_handle->connection_broken) {
    if (!request->use_polling_completion) pthread_mutex_unlock(&remote_thread_handle->mutex);
    TAKYON_RECORD_ERROR(path->error_message, "Remote side has failed\n");
    return false;
  }

  // Return results
  *bytes_received_ret = private_path->recv_completions[post_index].bytes_received;
  *piggyback_message_ret = private_path->recv_completions[post_index].piggyback_message;

  // Unlock
  if (!request->use_polling_completion) {
    pthread_mutex_unlock(&remote_thread_handle->mutex);
  }

  return true;
}
