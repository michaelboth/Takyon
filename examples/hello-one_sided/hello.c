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

static void writeMessage(TakyonPath *path, uint32_t i) {
  // Fill in the data to write
  char *message_addr = (char *)path->attrs.buffers[0].addr;
#ifdef ENABLE_CUDA
  char message_addr_cpu[TRANSPORT_BYTES];
  snprintf(message_addr_cpu, TRANSPORT_BYTES, "--- Iteration %u: Hello (CUDA) ---", i+1);
  uint64_t message_bytes = strlen(message_addr_cpu) + 1;
  cudaError_t cuda_status = cudaMemcpy(message_addr, message_addr_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
#else
  snprintf(message_addr, TRANSPORT_BYTES, "--- Iteration %u: Hello (CPU) ---", i+1);
  uint64_t message_bytes = strlen(message_addr) + 1;
#endif

  // Setup the one-sided write request
  TakyonOneSidedRequest write_request = { .is_write_request = true,
                                          .local_buffer = &path->attrs.buffers[0],
                                          .local_offset = 0,
                                          .remote_buffer_index = 1,
                                          .remote_offset = 0,
                                          .bytes = message_bytes,
                                          .use_is_done_notification = true,
                                          .use_polling_completion = false,
                                          .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  takyonOneSided(path, &write_request, TAKYON_WAIT_FOREVER, NULL);

  // If the interconnect supports non blocking transfers, then need to know when it's complete
  if (path->features.IsOneSidedDone_supported && write_request.use_is_done_notification) takyonIsOneSidedDone(path, &write_request, TAKYON_WAIT_FOREVER, NULL);
  printf("Message %d written\n", i+1);
}

static void readMessage(TakyonPath *path) {
  // Setup the one-sided write request
  TakyonOneSidedRequest read_request = { .is_write_request = false,
                                         .local_buffer = &path->attrs.buffers[2],
                                         .local_offset = 0,
                                         .remote_buffer_index = 1,
                                         .remote_offset = 0,
                                         .bytes = TRANSPORT_BYTES,
                                         .use_is_done_notification = true,
                                         .use_polling_completion = false,
                                         .usec_sleep_between_poll_attempts = 0 };

  // Start the read
  takyonOneSided(path, &read_request, TAKYON_WAIT_FOREVER, NULL);

  // If the interconnect supports non blocking transfers, then need to know when it's complete
  if (path->features.IsOneSidedDone_supported && read_request.use_is_done_notification) takyonIsOneSidedDone(path, &read_request, TAKYON_WAIT_FOREVER, NULL);

  // Process the data; i.e. print the received greeting
  char *message_addr = (char *)path->attrs.buffers[2].addr;
#ifdef ENABLE_CUDA
  char message_addr_cpu[TRANSPORT_BYTES];
  cudaError_t cuda_status = cudaMemcpy(message_addr_cpu, message_addr, TRANSPORT_BYTES, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
  printf("(CUDA): Read message '%s'\n", message_addr_cpu);
#else
  printf("(CPU): Read message '%s'\n", message_addr);
#endif
}

void hello(const bool is_endpointA, const char *interconnect, const uint32_t iterations) {
  printf("Hello Takyon Example (one-sided): endpoint %s: interconnect '%s'\n", is_endpointA ? "A" : "B", interconnect);

  // Create the memory buffers used with transfering data
  //   1. A writes from buffers[0] to buffer[1]
  //   2. A reads from buffers[1] to buffer[2]
  TakyonBuffer buffers[NUM_BUFFERS] = {};
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

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.interconnect, interconnect, TAKYON_MAX_INTERCONNECT_CHARS-1);
  attrs.is_endpointA                            = is_endpointA;
  attrs.failure_mode                            = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                               = TAKYON_VERBOSITY_ERRORS;
  attrs.buffer_count                            = NUM_BUFFERS;
  attrs.buffers                                 = buffers;
  attrs.max_pending_send_and_one_sided_requests = is_endpointA ? 1 : 0;
  attrs.max_pending_recv_requests               = 0;
  attrs.max_sub_buffers_per_send_request        = is_endpointA ? 1 : 0;
  attrs.max_sub_buffers_per_recv_request        = 0;

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 0, NULL, TAKYON_WAIT_FOREVER, &path);

  // Do the one-sided transfers, but only from endpoint A. B will not be involved
  if (path->attrs.is_endpointA) {
    for (uint32_t i=0; i<iterations; i++) {
      writeMessage(path, i);
      readMessage(path);
    }
  } else {
    printf("FYI: All the transfer activity is on endpoint A\n");
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