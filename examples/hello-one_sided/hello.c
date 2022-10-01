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
  #include "cuda.h"
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
#define MAX_TAKYON_BUFFERS 3

static uint64_t buildMessage(TakyonPath *path, uint32_t message_index) {
  // STEP 1: Get address for the message from the Takyon buffer
  char *message_addr = (char *)path->attrs.buffers[0].addr;

  // STEP 2: Fill in the message data
#ifdef ENABLE_CUDA
  // Cuda memory: first put in temporary CPU memory, then copy to CUDA
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  snprintf(message_addr_cpu, MAX_MESSAGE_BYTES, "--- Iteration %u: Hello (CUDA) ---", message_index+1);
  uint64_t message_bytes = strlen(message_addr_cpu) + 1;
  cudaError_t cuda_status = cudaMemcpy(message_addr, message_addr_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  // CPU memory
  snprintf(message_addr, MAX_MESSAGE_BYTES, "--- Iteration %u: Hello (CPU) ---", message_index+1);
  uint64_t message_bytes = strlen(message_addr) + 1;
#endif

  return message_bytes;
}

static void writeMessage(TakyonPath *path, uint64_t message_bytes, uint32_t message_index) {
  // STEP 1: Setup the Takyon sub buffer list (only one entry in the list)
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = 0 };

  // STEP 2: Setup the Takyon write request
  TakyonOneSidedRequest write_request = { .is_write_request = true,
                                          .sub_buffer_count = 1,
                                          .sub_buffers = &sub_buffer,
                                          .remote_buffer_index = 0, // This is not the same buffer as the local buffer at index 0
                                          .remote_offset = 0,
                                          .use_is_done_notification = true,
                                          .use_polling_completion = false,
                                          .usec_sleep_between_poll_attempts = 0 };

  // STEP 3: Start the Takyon one-sided write
  takyonOneSided(path, &write_request, TAKYON_WAIT_FOREVER, NULL);

  // STEP 4: If the Takyon Provider supports non-blocking writes, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported && write_request.use_is_done_notification) {
    takyonIsOneSidedDone(path, &write_request, TAKYON_WAIT_FOREVER, NULL);
  }

  printf("Message %d written\n", message_index+1);
}

static void *readMessage(TakyonPath *path) {
  // STEP 1: Setup the Takyon sub buffer list (only one entry in the list)
  TakyonSubBuffer sub_buffer = { .buffer_index = 1, .bytes = MAX_MESSAGE_BYTES, .offset = 0 };

  // STEP 2: Setup the Takyon read request
  TakyonOneSidedRequest read_request = { .is_write_request = false,
                                         .sub_buffer_count = 1,
                                         .sub_buffers = &sub_buffer,
                                         .remote_buffer_index = 0,
                                         .remote_offset = 0,
                                         .use_is_done_notification = true,
                                         .use_polling_completion = false,
                                         .usec_sleep_between_poll_attempts = 0 };

  // STEP 3: Start the Takyon one-sided read
  takyonOneSided(path, &read_request, TAKYON_WAIT_FOREVER, NULL);

  // STEP 4: If the Takyon Provider supports non-blocking reads, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported && read_request.use_is_done_notification) {
    takyonIsOneSidedDone(path, &read_request, TAKYON_WAIT_FOREVER, NULL);
  }

  return (char *)path->attrs.buffers[sub_buffer.buffer_index].addr + sub_buffer.offset;
}

static void processMessage(void *message_addr) {
#ifdef ENABLE_CUDA
  // CUDA memory: need to copy the GPU data to a temp CPU buffer
  char message_addr_cpu[MAX_MESSAGE_BYTES];
  cudaError_t cuda_status = cudaMemcpy(message_addr_cpu, message_addr, MAX_MESSAGE_BYTES, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
  printf("(CUDA): Read message '%s'\n", message_addr_cpu);
#else
  // CPU memory
  printf("(CPU): Read message '%s'\n", (char *)message_addr);
#endif
}

void hello(const bool is_endpointA, const char *provider, const uint32_t iterations) {
  printf("Hello Takyon Example (one-sided): endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);

  // Create the memory buffers used with transfering data
  // Endpoint 'A' need 2 buffer:
  //   Buffer 0: source message to be written to 'B'
  //   Buffer 1: destination message to be read from 'B'
  // Endpoint 'B' needs 1 buffer
  TakyonBuffer buffers[MAX_TAKYON_BUFFERS];
  uint32_t num_buffers = is_endpointA ? 2 : 1;
  for (uint32_t i=0; i<num_buffers; i++) {
    TakyonBuffer *buffer = &buffers[i];
    buffer->bytes = MAX_MESSAGE_BYTES;
    buffer->app_data = NULL;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer->addr, buffer->bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
    if (strncmp(provider, "Rdma", 4) == 0) {
      // Since this memory will transfer asynchronously via GPUDirect, need to mark the memory to be synchronous when accessing it after being received
      unsigned int flag = 1;
      CUresult cuda_result = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)buffer->addr);
      if (cuda_result != CUDA_SUCCESS) { printf("cuPointerSetAttribute() cuda_result: %d\n", cuda_result); exit(EXIT_FAILURE); }
    }
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

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                                   = is_endpointA;
  attrs.failure_mode                                   = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                                      = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                                   = num_buffers;
  attrs.buffers                                        = buffers;
  attrs.max_pending_send_and_one_sided_requests        = is_endpointA ? 1 : 0; // Only endpoint 'A' will be writing and reading
  attrs.max_pending_recv_requests                      = 0;                    // Endpoint 'B' isn't doing anyting transfers
  attrs.max_sub_buffers_per_send_and_one_sided_request = is_endpointA ? 1 : 0;
  attrs.max_sub_buffers_per_recv_request               = 0;

  // Create one side of the path: the other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 0, NULL, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers
  //   1. Endpoint 'A' writes from local buffers[0] to remote buffer[1]
  //   2. Endpoint 'A' reads from remote buffers[1] to local buffer[2]
  //   Endpoint 'B' is just not involved
  if (path->attrs.is_endpointA) {
    for (uint32_t i=0; i<iterations; i++) {
      uint64_t message_bytes = buildMessage(path, i);
      writeMessage(path, message_bytes, i);
      void *message_addr = readMessage(path);
      processMessage(message_addr);
    }
  } else {
    printf("FYI: All the transfer activity is on endpoint A\n");
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);

  // Free the takyon buffers
  for (uint32_t i=0; i<num_buffers; i++) {
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
