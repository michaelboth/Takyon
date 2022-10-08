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

#include "takyon.h"
#include "hello.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef ENABLE_CUDA
  #ifndef _WIN32
    #include "cuda.h"
  #endif
  #include "cuda_runtime.h"
#endif
#ifdef ENABLE_MMAP
  #include "utils_ipc.h"
#endif
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

#define MAX_MESSAGE_BYTES 100 // Just need enough to transfer a nice text greeting

static uint64_t buildMessage(TakyonPath *path, uint32_t message_index) {
  // STEP 1: Get address for the message from the Takyon buffer
  char *message_addr = (char *)path->attrs.buffers[0].addr;
  const char *endpoint_text = path->attrs.is_endpointA ? "A" : "B";

  // STEP 2: Fill in the message data
#ifdef ENABLE_CUDA
  // Cuda memory: first put in temporary CPU memory, then copy to CUDA
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  snprintf(message_addr_cpu, MAX_MESSAGE_BYTES, "Hello %u from %s", message_index+1, endpoint_text);
  printf("  %s (CUDA): \"%s\"\n", endpoint_text, message_addr_cpu);
  uint64_t message_bytes = strlen(message_addr_cpu) + 1;
  cudaError_t cuda_status = cudaMemcpy(message_addr, message_addr_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  // CPU memory
  snprintf(message_addr, MAX_MESSAGE_BYTES, "Hello %d from %s", message_index+1, endpoint_text);
  printf("  %s (CPU): \"%s\"\n", endpoint_text, message_addr);
  uint64_t message_bytes = strlen(message_addr) + 1;
#endif

  return message_bytes;
}

static void writeMessage(TakyonPath *path, uint64_t message_bytes) {
  // STEP 1: Setup the Takyon sub buffer list (only one entry in the list)
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = 0 };

  // STEP 2: Setup the Takyon write request
  TakyonOneSidedRequest write_request = { .operation = TAKYON_OP_WRITE,
                                          .sub_buffer_count = 1,
                                          .sub_buffers = &sub_buffer,
                                          .remote_buffer_index = 0, // This is not the same buffer as the local buffer at index 0
                                          .remote_offset = 0,
                                          .submit_fence = false,
                                          .use_is_done_notification = true,
                                          .use_polling_completion = false,
                                          .usec_sleep_between_poll_attempts = 0 };

  // STEP 3: Start the Takyon one-sided write
  takyonOneSided(path, &write_request, TAKYON_WAIT_FOREVER, NULL);

  // STEP 4: If the Takyon Provider supports non-blocking writes, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_function_supported && write_request.use_is_done_notification) {
    takyonIsOneSidedDone(path, &write_request, TAKYON_WAIT_FOREVER, NULL);
  }
}

static void *readMessage(TakyonPath *path) {
  // STEP 1: Setup the Takyon sub buffer list (only one entry in the list)
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = MAX_MESSAGE_BYTES, .offset = 0 };

  // STEP 2: Setup the Takyon read request
  TakyonOneSidedRequest read_request = { .operation = TAKYON_OP_READ,
                                         .sub_buffer_count = 1,
                                         .sub_buffers = &sub_buffer,
                                         .remote_buffer_index = 0,
                                         .remote_offset = 0,
                                         .submit_fence = false,
                                         .use_is_done_notification = true,
                                         .use_polling_completion = false,
                                         .usec_sleep_between_poll_attempts = 0 };

  // STEP 3: Start the Takyon one-sided read
  takyonOneSided(path, &read_request, TAKYON_WAIT_FOREVER, NULL);

  // STEP 4: If the Takyon Provider supports non-blocking reads, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_function_supported && read_request.use_is_done_notification) {
    takyonIsOneSidedDone(path, &read_request, TAKYON_WAIT_FOREVER, NULL);
  }

  return (char *)path->attrs.buffers[sub_buffer.buffer_index].addr + sub_buffer.offset;
}

static void processMessage(char *message_addr) {
#ifdef ENABLE_CUDA
  // CUDA memory: need to copy the GPU data to a temp CPU buffer
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  cudaError_t cuda_status = cudaMemcpy(message_addr_cpu, message_addr, MAX_MESSAGE_BYTES, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
  printf("  B (CUDA): \"%s\"\n", message_addr_cpu);
#else
  // CPU memory
  printf("  B (CPU): \"%s\"\n", message_addr);
#endif
}

static void sendSignal(TakyonPath *path) {
  // Setup the send request
  TakyonSendRequest send_request = { .sub_buffer_count = 0,
                                     .sub_buffers = NULL,
                                     .submit_fence = false,
                                     .use_is_sent_notification = true,
                                     .use_polling_completion = false,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggyback_message = 0;
  takyonSend(path, &send_request, piggyback_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_function_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void recvSignal(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for signal to arrive
  takyonIsRecved(path, recv_request, TAKYON_WAIT_FOREVER, NULL, NULL, NULL);

  // If the provider supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->capabilities.PostRecvs_function_supported) takyonPostRecvs(path, 1, recv_request);
}

void hello(const bool is_endpointA, const char *provider, const uint32_t iterations) {
  printf("Hello Takyon Example (one-sided): endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);

  // Create the memory buffer used with transfering messages
  // Only one buffer on each endpoint
  TakyonBuffer buffer;
  {
    buffer.bytes = MAX_MESSAGE_BYTES;
    buffer.app_data = NULL;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer.addr, buffer.bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#ifndef _WIN32
    if (strncmp(provider, "Rdma", 4) == 0) {
      // Since this memory will transfer asynchronously via GPUDirect, need to mark the memory to be synchronous when accessing it after being received
      unsigned int flag = 1;
      CUresult cuda_result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buffer.addr);
      if (cuda_result != CUDA_SUCCESS) { printf("cuPointerSetAttribute() cuda_result: %d\n", cuda_result); exit(EXIT_FAILURE); }
    }
#endif
#else
#ifdef ENABLE_MMAP
    if (strncmp(provider, "InterProcess", 12) == 0) {
      snprintf(buffer.name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_hello_buffer_" UINT64_FORMAT, is_endpointA ? "A" : "B", buffer.bytes);
      char error_message[300];
      bool ok = mmapAlloc(buffer.name, buffer.bytes, &buffer.addr, &buffer.app_data, error_message, 300);
      if (!ok) { printf("mmapAlloc() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      buffer.addr = malloc(buffer.bytes);
    }
#else
    buffer.addr = malloc(buffer.bytes);
#endif
#endif
  }

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                          = is_endpointA;
  attrs.failure_mode                          = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                             = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                          = 1;
  attrs.buffers                               = &buffer;
  attrs.max_pending_send_requests             = 1; // Only used for signaling; no data transfers
  attrs.max_pending_recv_requests             = 1; // Only used for signaling; no data transfers
  attrs.max_pending_one_sided_requests        = 1; // 'A' will 'write', 'B' will 'read'
  attrs.max_sub_buffers_per_send_request      = 0;
  attrs.max_sub_buffers_per_recv_request      = 0;
  attrs.max_sub_buffers_per_one_sided_request = 1;

  // Setup the initial recv request used for signaling
  TakyonRecvRequest recv_request;
  recv_request.sub_buffer_count = 0;
  recv_request.sub_buffers = NULL;
  recv_request.use_polling_completion = false;
  recv_request.usec_sleep_between_poll_attempts = 0;

  // Create one side of the path: the other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 1, &recv_request, TAKYON_WAIT_FOREVER, &path);

  // One-sided reliable 'write'
  if (!path->capabilities.is_unreliable && path->capabilities.one_sided_write_supported) {
    printf("%s: Testing one-sided reliable 'write' via endpoint A\n", path->attrs.is_endpointA ? "A" : "B");
    for (uint32_t i=0; i<iterations; i++) {
      if (path->attrs.is_endpointA) {
        // Fill in message
        uint64_t message_bytes = buildMessage(path, i);
        // One-sided 'write' from local buffers[0] to remote buffer[0] (remote endpoint is not involved with this transfer)
        writeMessage(path, message_bytes);
        // Send a signal to let the remote endpoint know the message has arrived
        sendSignal(path);
        // Wait for the remote endpoint to signal back so it safe to send again
        recvSignal(path, &recv_request);
      } else {
        // Wait for the signal to know the message has arrived (via a 'write' transfer)
        recvSignal(path, &recv_request);
        // Process the message
        char *message_addr = (char *)path->attrs.buffers[0].addr;
        processMessage(message_addr);
        // Send a signal to let the remote endpoint know another message can be 'written'
        sendSignal(path);
      }
    }
  }

  // One-sided reliable 'read'
  if (!path->capabilities.is_unreliable && path->capabilities.one_sided_read_supported) {
    printf("%s: Testing one-sided reliable 'read' via endpoint B\n", path->attrs.is_endpointA ? "A" : "B");
    for (uint32_t i=0; i<iterations; i++) {
      if (path->attrs.is_endpointA) {
        // Fill in message
        (void)buildMessage(path, i);
        // Send a signal to let the remote endpoint know it can 'read' the message
        sendSignal(path);
        // Wait for a signal from the remote endpoint to know another message can be prepared
        recvSignal(path, &recv_request);
      } else {
        // Wait for a signal from the remote endpoint to know the message has been prepared
        recvSignal(path, &recv_request);
        // Read the message
        void *message_addr = readMessage(path);
        // Process the message
        processMessage(message_addr);
        // Send a signal to let the remote endpoint know it can prepare the next message
        sendSignal(path);
      }
    }
  }

  // One-sided reliable 'atomics'
  /*+ atomics
  if (!path->capabilities.is_unreliable && path->capabilities.one_sided_atomics_supported) {
    printf("%s: Testing one-sided reliable 'atomics' via endpoint A\n", path->attrs.is_endpointA ? "A" : "B");
    for (uint32_t i=0; i<iterations; i++) {
      if (path->attrs.is_endpointA) {
      } else {
      }
    }
  }
  */

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);

  // Free the takyon buffer resources
  {
#ifdef ENABLE_CUDA
    cudaFree(buffer.addr);
#else
#ifdef ENABLE_MMAP
    if (buffer.app_data != NULL) {
      char error_message[300];
      bool ok = mmapFree(buffer.app_data, error_message, 300);
      if (!ok) { printf("mmapFree() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      free(buffer.addr);
    }
#else
    free(buffer.addr);
#endif
#endif
  }

  printf("Hello Takyon Example: %s is done.\n", is_endpointA ? "A" : "B");
}
