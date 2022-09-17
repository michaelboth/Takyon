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

#define TRANSPORT_BYTES 100 // Just need enough to transfer a nice text greeting
#define NUM_BUFFERS 3

static void sendMessage(TakyonPath *path, uint32_t i) {
  // Fill in the data to send
  //   - If multiple sub buffers are supported by the interconect, this send request
  //     will use two separate sub buffers to send the single text greeting: the
  //     first 10 characters will be in the second sub buffers, and the remaining
  //     part of the message will be in the first sub buffer.
  //   - If one one sub buffer is supported, then the greeting will be in a single
  //     sub buffer
  //   - Receiver does not need to match the sender's sub buffer count
  char *message_addr1 = (char *)path->attrs.buffers[0].addr;
  char *message_addr2 = (char *)path->attrs.buffers[1].addr;
  uint64_t split_bytes = 10;
#ifdef ENABLE_CUDA
  char message_addr_cpu[TRANSPORT_BYTES];
  snprintf(message_addr_cpu, TRANSPORT_BYTES, "--- Iteration %u: Hello from %s (CUDA, %d %s) ---", i+1, path->attrs.is_endpointA ? "A" : "B",
           path->features.multi_sub_buffers_supported ? 2 : 1, path->features.multi_sub_buffers_supported ? "sub buffers" : "sub buffer");
  uint64_t message_bytes = strlen(message_addr_cpu) + 1;
  cudaError_t cuda_status = cudaMemcpy(message_addr1, message_addr_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
  if (path->features.multi_sub_buffers_supported) {
    // Multiple sub buffers are supported: copy the beginning of the message to second buffer
    cuda_status = cudaMemcpy(message_addr2, message_addr1, split_bytes, cudaMemcpyDefault);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
  }
#else
  snprintf(message_addr1, TRANSPORT_BYTES, "--- Iteration %u: Hello from %s (CPU, %d %s) ---", i+1, path->attrs.is_endpointA ? "A" : "B",
           path->features.multi_sub_buffers_supported ? 2 : 1, path->features.multi_sub_buffers_supported ? "sub buffers" : "sub buffer");
  uint64_t message_bytes = strlen(message_addr1) + 1;
  if (path->features.multi_sub_buffers_supported) {
    // Multiple sub buffers are supported: copy the beginning of the message to second buffer
    memcpy(message_addr2, message_addr1, split_bytes);
    memset(message_addr1, 'x', split_bytes); // Clear the unsent bytes to prove multi sub buffers is working
  }
#endif

  // Setup the send request
  //  - Initially defined with 2 sub buffers, but will only use the first if multiple sub buffers are not supported
  TakyonSubBuffer sender_sub_buffers[2] = {{ .buffer = &path->attrs.buffers[1], .bytes = split_bytes, .offset = 0 },
                                           { .buffer = &path->attrs.buffers[0], .bytes = message_bytes-split_bytes, .offset = split_bytes }};
  if (!path->features.multi_sub_buffers_supported) {
    // Interconnect only supports a single sub buffer
    sender_sub_buffers[0].buffer = &path->attrs.buffers[0];
    sender_sub_buffers[0].bytes = message_bytes;
    sender_sub_buffers[0].offset = 0;
  }
  TakyonSendRequest send_request = { .sub_buffer_count = path->features.multi_sub_buffers_supported ? 2 : 1,
                                     .sub_buffers = sender_sub_buffers,
                                     .use_is_sent_notification = true,
                                     .use_polling_completion = false,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = (path->features.piggy_back_message_supported) ? i : 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);
  if (!path->features.IsRecved_supported) {
    printf("Message %d sent (one way, %d %s)\n", i+1, path->features.multi_sub_buffers_supported ? 2 : 1, path->features.multi_sub_buffers_supported ? "sub buffers" : "sub buffer");
  }

  // If the interconnect supports non blocking sends, then need to know when it's complete
  if (path->features.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for data to arrive
  //   - Recv request only has a single sub buffer, regardless of what the sender has
  uint64_t bytes_received;
  uint32_t piggy_back_message;
  takyonIsRecved(path, recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, &piggy_back_message);

  // Process the data; i.e. print the received greeting
  TakyonSubBuffer *recver_sub_buffer = &recv_request->sub_buffers[0];
  char *message_addr = (char *)recver_sub_buffer->buffer->addr + recver_sub_buffer->offset;
#ifdef ENABLE_CUDA
  char message_addr_cpu[TRANSPORT_BYTES];
  cudaError_t cuda_status = cudaMemcpy(message_addr_cpu, message_addr, bytes_received, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
  if (path->features.piggy_back_message_supported) {
    printf("%s (CUDA): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=%u\n", path->attrs.is_endpointA ? "A" : "B", message_addr_cpu, bytes_received, piggy_back_message);
  } else {
    printf("%s (CUDA): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=NOT SUPPORTED\n", path->attrs.is_endpointA ? "A" : "B", message_addr_cpu, bytes_received);
  }
#else
  if (path->features.piggy_back_message_supported) {
    printf("%s (CPU): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=%u\n", path->attrs.is_endpointA ? "A" : "B", message_addr, bytes_received, piggy_back_message);
  } else {
    printf("%s (CPU): Got message '%s', bytes=" UINT64_FORMAT ", piggy_back_message=NOT SUPPORTED\n", path->attrs.is_endpointA ? "A" : "B", message_addr, bytes_received);
  }
#endif

  // If the interconnect supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->features.PostRecvs_supported) takyonPostRecvs(path, 1, recv_request);
}

void hello(const bool is_endpointA, const char *interconnect, const uint32_t iterations) {
  printf("Hello Takyon Example (two-sided): endpoint %s: interconnect '%s'\n", is_endpointA ? "A" : "B", interconnect);

  // Create the memory buffers used with transfering data
  //   - The first 2 are for the sender, and the 3rd is for the receiver
  TakyonBuffer buffers[NUM_BUFFERS];
  for (uint32_t i=0; i<NUM_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
    buffer->bytes = TRANSPORT_BYTES;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer->addr, buffer->bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
#else
#ifdef ENABLE_MMAP
    if (strncmp(interconnect, "InterProcess ", 13) == 0) {
      snprintf(buffer->name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_hello_buffer_%d_" UINT64_FORMAT, is_endpointA ? "A" : "B", i, buffer->bytes);
      char error_message[300];
      bool ok = mmapAlloc(buffer->name, buffer->bytes, &buffer->addr, &buffer->app_data, error_message, 300);
      if (!ok) { printf("mmapAlloc() failed: %s\n", error_message); exit(0); }
    } else {
      buffer->addr = malloc(buffer->bytes);
    }
#else
    buffer->addr = malloc(buffer->bytes);
#endif
#endif
  }

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recieves posted before sending can start
  TakyonSubBuffer recver_sub_buffer = { .buffer = &buffers[2], .bytes = TRANSPORT_BYTES, .offset = 0 };
  TakyonRecvRequest recv_request = { .sub_buffer_count = 1,
                                     .sub_buffers = &recver_sub_buffer,
                                     .use_polling_completion = false,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.interconnect, interconnect, TAKYON_MAX_INTERCONNECT_CHARS-1);
  attrs.is_endpointA                            = is_endpointA;
  attrs.failure_mode                            = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                               = TAKYON_VERBOSITY_ERRORS;
  attrs.buffer_count                            = NUM_BUFFERS;
  attrs.buffers                                 = buffers;
  attrs.max_pending_send_and_one_sided_requests = 1;
  attrs.max_pending_recv_requests               = 1;
  attrs.max_sub_buffers_per_send_request        = 2; // Will send two blocks if supported, otherwise one block.
  attrs.max_sub_buffers_per_recv_request        = 1; // Recevier will always get a single block

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 1, &recv_request, TAKYON_WAIT_FOREVER, &path);

  // Take turns sending the greeing multiple times
  //  - If this is a connection oriented communication, then messages are sent in both directions and are self synchronizing to avoid race conditions
  //  - If this is NOT a connection oriented communication, then messages are only sent from A to B and will be dropped if B is not running before the sender
  for (uint32_t i=0; i<iterations; i++) {
    if (path->attrs.is_endpointA) {
      // Send message
      sendMessage(path, i);
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      if (path->features.IsRecved_supported) recvMessage(path, &recv_request);
    } else {
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      recvMessage(path, &recv_request);
      // Send message
      if (path->features.Send_supported) sendMessage(path, i);
    }
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);

  // Free the takyon buffers
  for (uint32_t i=0; i<NUM_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
#ifdef ENABLE_CUDA
    cudaFree(buffer->addr);
#else
#ifdef ENABLE_MMAP
    if (buffer->app_data != NULL) {
      char error_message[300];
      bool ok = mmapFree(buffer->app_data, error_message, 300);
      if (!ok) { printf("mmapFree() failed: %s\n", error_message); exit(0); }
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
