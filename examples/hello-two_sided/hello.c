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

#define NUM_TAKYON_BUFFERS 3
#define MAX_MESSAGE_BYTES 100
#define MESSAGE_SPLIT_BYTES 10
#define FIRST_RECV_TIMEOUT_SECONDS 5.0    // Wait longer regardless if the connection is reliable or unreliable
#define ACTIVE_RECV_TIMEOUT_SECONDS 0.25  // After the first message is received, don't want to sit around waiting if the connection is unreliable

static uint64_t buildMultiBufferMessage(TakyonPath *path, uint32_t message_index) {
  // STEP 1: Get addresses for the multi-buffer message
  char *message_addr1 = (char *)path->attrs.buffers[0].addr;
  char *message_addr2 = (char *)path->attrs.buffers[1].addr; // Won't be used if Provider doesn't support multi-buffers

  // STEP 2: Fill in the message data (the entire greeting) in the first Takyon buffer
#ifdef ENABLE_CUDA
  // Cuda memory: first put in temporary CPU memory, then copy to CUDA
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  snprintf(message_addr_cpu, MAX_MESSAGE_BYTES, "--- Iteration %u: Hello from %s (CUDA, %d %s) ---", message_index+1, path->attrs.is_endpointA ? "A" : "B",
           path->capabilities.multi_sub_buffers_supported ? 2 : 1, path->capabilities.multi_sub_buffers_supported ? "sub buffers" : "sub buffer");
  uint64_t message_bytes = strlen(message_addr_cpu) + 1;
  cudaError_t cuda_status = cudaMemcpy(message_addr1, message_addr_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  // CPU memory
  snprintf(message_addr1, MAX_MESSAGE_BYTES, "--- Iteration %u: Hello from %s (CPU, %d %s) ---", message_index+1, path->attrs.is_endpointA ? "A" : "B",
           path->capabilities.multi_sub_buffers_supported ? 2 : 1, path->capabilities.multi_sub_buffers_supported ? "sub buffers" : "sub buffer");
  uint64_t message_bytes = strlen(message_addr1) + 1;
#endif

  // STEP 3: If transferring multi buffers is supported, then split the message up into two Takyon buffers
  if (path->capabilities.multi_sub_buffers_supported) {
#ifdef ENABLE_CUDA
    cuda_status = cudaMemcpy(message_addr2, message_addr1, MESSAGE_SPLIT_BYTES, cudaMemcpyDefault);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
    // Clear the unsent bytes to prove multi sub buffers is working
    memset(message_addr_cpu, 'x', MESSAGE_SPLIT_BYTES);
    cudaError_t cuda_status = cudaMemcpy(message_addr1, message_addr_cpu, MESSAGE_SPLIT_BYTES, cudaMemcpyDefault);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
    memcpy(message_addr2, message_addr1, MESSAGE_SPLIT_BYTES);
    // Clear the unsent bytes to prove multi sub buffers is working
    memset(message_addr1, 'x', MESSAGE_SPLIT_BYTES);
#endif
  }

  return message_bytes;
}

static void sendMessage(TakyonPath *path, uint64_t message_bytes, uint32_t message_index) {
  // STEP 1: Setup the Takyon sub buffer list assuming multi-buffers are supported
  TakyonSubBuffer sender_sub_buffers[2] = {{ .buffer_index = 1, .bytes = MESSAGE_SPLIT_BYTES, .offset = 0 },
                                           { .buffer_index = 0, .bytes = message_bytes-MESSAGE_SPLIT_BYTES, .offset = MESSAGE_SPLIT_BYTES }};
  // See if the provider only supports a single sub buffer
  if (!path->capabilities.multi_sub_buffers_supported) {
    sender_sub_buffers[0].buffer_index = 0;
    sender_sub_buffers[0].bytes = message_bytes;
    sender_sub_buffers[0].offset = 0;
  }

  // STEP 2: Setup the Takyon send request
  TakyonSendRequest send_request = { .sub_buffer_count = path->capabilities.multi_sub_buffers_supported ? 2 : 1,
                                     .sub_buffers = sender_sub_buffers,
                                     .use_is_sent_notification = true,
				     .use_polling_completion = false,
                                     .usec_sleep_between_poll_attempts = 0 };

  // STEP 3: Start the Takyon send
  uint32_t piggy_back_message = (path->capabilities.piggy_back_messages_supported) ? message_index : 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);
  if (path->capabilities.is_unreliable) {
    printf("Message %d sent (%d %s, " UINT64_FORMAT " bytes)\n", message_index+1, path->capabilities.multi_sub_buffers_supported ? 2 : 1, path->capabilities.multi_sub_buffers_supported ? "sub buffers" : "sub buffer", message_bytes);
  }

  // STEP 4: If the Takyon Provider supports non-blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_supported && send_request.use_is_sent_notification) {
    takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
  }
}

static void recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request, double timeout, uint64_t *bytes_received_out, uint32_t *piggy_back_message_out) {
  // Wait for data to arrive
  // NOTES:
  //  - The recv request was already setup before the Takyon path was created
  //  - The recv request only has a single sub buffer, regardless of what the sender has
  bool timed_out;
  takyonIsRecved(path, recv_request, timeout, &timed_out, bytes_received_out, piggy_back_message_out);
  if (timed_out)  { printf("\nTimed out waiting for messages\n"); exit(EXIT_SUCCESS); }
}

static void processSingleBufferMessage(TakyonPath *path, TakyonRecvRequest *recv_request, bool is_rdma_UD, uint64_t bytes_received, uint32_t piggy_back_message) {
  // STEP 1: Get the message address
  TakyonSubBuffer *recver_sub_buffer = &recv_request->sub_buffers[0];
  char *message_addr = (char *)path->attrs.buffers[recver_sub_buffer->buffer_index].addr + recver_sub_buffer->offset;
  // IMPORTANT: if this is an RDMA UD (unreliable datagram) transfer, the first 40 bytes will contain the RDMA's Global Routing Header. Need to skip over this!
  if (is_rdma_UD) {
    message_addr += 40;
    bytes_received -= 40;
  }

  // STEP 2: Print the greeting contained in the Takyon buffer
#ifdef ENABLE_CUDA
  // CUDA memory: need to copy the GPU data to a temp CPU buffer
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  cudaError_t cuda_status = cudaMemcpy(message_addr_cpu, message_addr, bytes_received, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
  if (path->capabilities.piggy_back_messages_supported) {
    printf("%s (CUDA): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=%u\n", path->attrs.is_endpointA ? "A" : "B", message_addr_cpu, bytes_received, piggy_back_message);
  } else {
    printf("%s (CUDA): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=NOT SUPPORTED\n", path->attrs.is_endpointA ? "A" : "B", message_addr_cpu, bytes_received);
  }
#else
  // CPU memory
  if (path->capabilities.piggy_back_messages_supported) {
    printf("%s (CPU): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=%u\n", path->attrs.is_endpointA ? "A" : "B", message_addr, bytes_received, piggy_back_message);
  } else {
    printf("%s (CPU): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=NOT SUPPORTED\n", path->attrs.is_endpointA ? "A" : "B", message_addr, bytes_received);
  }
#endif
}

void hello(const bool is_endpointA, const char *provider, const uint32_t iterations) {
  printf("Hello Takyon Example (two-sided): endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);

  // Create the memory buffers used with transfering data
  //   - The first 2 are for the sender, and the 3rd is for the receiver
  //   - If the provider is RDMA UD (unreliable datagram), then receiver needs to allocate 40 extra bytes for RDMA's global routing header
  bool is_rdma_UD = (strncmp(provider, "RdmaUD", 6) == 0);
  TakyonBuffer buffers[NUM_TAKYON_BUFFERS];
  for (uint32_t i=0; i<NUM_TAKYON_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
    buffer->bytes = MAX_MESSAGE_BYTES + (is_rdma_UD ? 40 : 0);
    buffer->app_data = NULL;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer->addr, buffer->bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#ifndef _WIN32
    if (strncmp(provider, "Rdma", 4) == 0) {
      // Since this memory will transfer asynchronously via GPUDirect, need to mark the memory to be synchronous when accessing it after being received
      unsigned int flag = 1;
      CUresult cuda_result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buffer->addr);
      if (cuda_result != CUDA_SUCCESS) { printf("cuPointerSetAttribute() cuda_result: %d\n", cuda_result); exit(EXIT_FAILURE); }
    }
#endif
#else
#ifdef ENABLE_MMAP
    if (strncmp(provider, "InterProcess", 12) == 0) {
      snprintf(buffer->name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_hello_buffer_%d_" UINT64_FORMAT, is_endpointA ? "A" : "B", i, buffer->bytes);
      char error_message[300];
      bool ok = mmapAlloc(buffer->name, buffer->bytes, &buffer->addr, &buffer->app_data, error_message, 300);
      if (!ok) { printf("mmapAlloc() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      buffer->addr = malloc(buffer->bytes);
    }
#else
    buffer->addr = malloc(buffer->bytes);
#endif
#endif
  }

  // Define the path attributes; can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                                   = is_endpointA;
  attrs.failure_mode                                   = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                                      = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                                   = NUM_TAKYON_BUFFERS;
  attrs.buffers                                        = buffers;
  attrs.max_pending_send_and_one_sided_requests        = 1; // Only one message will be in transit at any point in time
  attrs.max_pending_recv_requests                      = 1; // Only one recv request will ever be posted at any point in time
  attrs.max_sub_buffers_per_send_and_one_sided_request = 2; // Will send two blocks if supported, otherwise only one block will be used
  attrs.max_sub_buffers_per_recv_request               = 1; // Receiver will always get a single block

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recieves posted before sending can start
  TakyonSubBuffer recver_sub_buffer = { .buffer_index = 2, .bytes = MAX_MESSAGE_BYTES + (is_rdma_UD ? 40 : 0), .offset = 0 };
  TakyonRecvRequest recv_request = { .sub_buffer_count = 1,
                                     .sub_buffers = &recver_sub_buffer,
				     .use_polling_completion = false,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Create one side of the path; the other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 1, &recv_request, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers
  //  - If this is a reliable connection, then messages are sent in both directions and are self synchronizing to avoid race conditions
  //  - If this is an unrelaible connection, then messages are only sent from A to B and will be dropped if B doesn't re-post the recv before the message arrives
  for (uint32_t i=0; i<iterations; i++) {
    if (path->attrs.is_endpointA) {
      // Prepare the message and send it
      uint64_t message_bytes = buildMultiBufferMessage(path, i);
      sendMessage(path, message_bytes, i);

      if (!path->capabilities.is_unreliable) {
        // Recv the message, process it, then re-post the recv requewst
        uint64_t bytes_received;
        uint32_t piggy_back_message;
        recvMessage(path, &recv_request, ACTIVE_RECV_TIMEOUT_SECONDS, &bytes_received, &piggy_back_message);
        processSingleBufferMessage(path, &recv_request, is_rdma_UD, bytes_received, piggy_back_message);
        if (path->capabilities.PostRecvs_supported) { takyonPostRecvs(path, 1, &recv_request); }
      }

    } else {
      // Recv the message, process it, then re-post the recv requewst
      uint64_t bytes_received;
      uint32_t piggy_back_message;
      double timeout = (i==0) ? FIRST_RECV_TIMEOUT_SECONDS : ACTIVE_RECV_TIMEOUT_SECONDS;
      recvMessage(path, &recv_request, timeout, &bytes_received, &piggy_back_message);
      processSingleBufferMessage(path, &recv_request, is_rdma_UD, bytes_received, piggy_back_message);
      if (path->capabilities.PostRecvs_supported) { takyonPostRecvs(path, 1, &recv_request); }

      if (!path->capabilities.is_unreliable) {
        // Prepare the message and send it
        uint64_t message_bytes = buildMultiBufferMessage(path, i);
        sendMessage(path, message_bytes, i);
      }
    }
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);

  // Free the takyon buffers
  for (uint32_t i=0; i<NUM_TAKYON_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
#ifdef ENABLE_CUDA
    cudaFree(buffer->addr);
#else
#ifdef ENABLE_MMAP
    if (buffer->app_data != NULL) {
      char error_message[300];
      bool ok = mmapFree(buffer->app_data, error_message, 300);
      if (!ok) { printf("mmapFree() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      free(buffer->addr);
    }
#else
    free(buffer->addr);
#endif
#endif
  }

  printf("Hello Takyon Example: %s is done.\n", is_endpointA ? "A" : "B");
}
