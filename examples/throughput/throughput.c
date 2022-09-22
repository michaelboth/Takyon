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
#include "utils_time.h"
#include "throughput.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
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

#define NUM_TAKYON_BUFFERS 2
#define RECV_TIMEOUT_SECONDS 5.0

static uint64_t L_detected_drops = 0;

static void fillInMessage(TakyonPath *path, const uint64_t message_bytes, const uint64_t message_count) {
#ifdef ENABLE_CUDA
  uint64_t *data_cpu = path->attrs.buffers[0].app_data;
  uint64_t *data_gpu = path->attrs.buffers[0].addr;
#else
  uint64_t *data_cpu = path->attrs.buffers[0].addr;
#endif
  uint64_t elements = message_bytes / sizeof(uint64_t);
  for (uint64_t i=0; i<elements; i++) {
    data_cpu[i] = i + message_count;
  }
#ifdef ENABLE_CUDA
  cudaError_t cuda_status = cudaMemcpy(data_gpu, data_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#endif
}

static void validateMessage(TakyonBuffer *buffer, const uint64_t bytes_received, const uint64_t message_bytes, const uint64_t message_offset, const uint64_t message_count, uint64_t *previous_start_value_inout) {
#ifdef ENABLE_CUDA
  uint64_t *data_cpu = buffer->app_data;
  uint64_t *data_gpu = (uint64_t *)((uint8_t *)buffer->addr + message_offset);
  cudaError_t cuda_status = cudaMemcpy(data_cpu, data_gpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  uint64_t *data_cpu = (uint64_t *)((uint8_t *)buffer->addr + message_offset);
#endif
  if (bytes_received != message_bytes) { printf("Message " UINT64_FORMAT ": Received " UINT64_FORMAT " bytes, but expect " UINT64_FORMAT " bytes\n", message_count, bytes_received, message_bytes); exit(EXIT_FAILURE); }
  uint64_t previous_start_value = *previous_start_value_inout;
  if (previous_start_value >= data_cpu[0]) { printf("Message " UINT64_FORMAT ": Message start value=" UINT64_FORMAT " did not increase from previous message value=" UINT64_FORMAT ". Problem with provider?\n", message_count, data_cpu[0], previous_start_value); exit(EXIT_FAILURE); }
  uint64_t elements = message_bytes / sizeof(uint64_t);
  for (uint64_t i=1; i<elements; i++) {
    if ((data_cpu[i-1]+1) != data_cpu[i]) { printf("Message " UINT64_FORMAT ": data[" UINT64_FORMAT "]=" UINT64_FORMAT " and data[" UINT64_FORMAT "]=" UINT64_FORMAT " did not increase by 1\n", message_count, i-1, data_cpu[i-1], i, data_cpu[i]); exit(EXIT_FAILURE); }
  }
  // Count drops
  L_detected_drops += data_cpu[0] - (previous_start_value+1);
  *previous_start_value_inout = data_cpu[0];
}

static void sendMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint64_t message_count) {
  if (validate) {
    fillInMessage(path, message_bytes, message_count);
  }

  // Setup the send request
  TakyonSubBuffer sender_sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = 0 };
  TakyonSendRequest send_request = { .sub_buffer_count = 1,
                                     .sub_buffers = &sender_sub_buffer,
                                     .use_is_sent_notification = true, /*+ test without */
                                     .use_polling_completion = use_polling_completion,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request, const bool validate, const uint64_t message_count) {
  // Wait for data to arrive
  uint64_t bytes_received;
  bool timed_out;
  takyonIsRecved(path, recv_request, RECV_TIMEOUT_SECONDS, &timed_out, &bytes_received, NULL);
  if (timed_out)  { printf("Timed out waiting for a message\n"); exit(EXIT_FAILURE); }
  assert(bytes_received == recv_request->sub_buffers[0].bytes);

  if (validate) {
    static uint64_t previous_start_value = 0;
    uint64_t message_bytes = recv_request->sub_buffers[0].bytes;
    uint64_t message_offset = recv_request->sub_buffers[0].offset;
    TakyonBuffer *buffer = &path->attrs.buffers[recv_request->sub_buffers[0].buffer_index];
    validateMessage(buffer, bytes_received, message_bytes, message_offset, message_count, &previous_start_value);
  }
}

static void writeMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint64_t message_count) {
  if (validate) {
    fillInMessage(path, message_bytes, message_count);
  }

  // Setup the one-sided write request
  TakyonOneSidedRequest request = {
                                   .is_write_request = true,
                                   .local_buffer_index = 0,
                                   .local_offset = 0,
                                   .remote_buffer_index = 1,
                                   .remote_offset = 0,
                                   .bytes = message_bytes,
                                   .use_is_done_notification = true, /*+ test without */
                                   .use_polling_completion = use_polling_completion,
                                   .usec_sleep_between_poll_attempts = 0
  };

  // Start the write
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported && request.use_is_done_notification) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);
}

static void readMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint64_t message_count) {
  // Setup the one-sided write request
  uint64_t message_offset = message_bytes;
  TakyonOneSidedRequest request = {
                                   .is_write_request = false,
                                   .local_buffer_index = 0,
                                   .local_offset = message_offset, /*+ make sure have space for two messages */
                                   .remote_buffer_index = 1,
                                   .remote_offset = 0,
                                   .bytes = message_bytes,
                                   .use_is_done_notification = true,
                                   .use_polling_completion = use_polling_completion,
                                   .usec_sleep_between_poll_attempts = 0
  };

  // Start the write
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);

  if (validate) {
    static uint64_t previous_start_value = 0;
    TakyonBuffer *buffer = &path->attrs.buffers[0];
    validateMessage(buffer, message_bytes, message_bytes, message_offset, message_count, &previous_start_value);
  }
}

static void sendSignal(TakyonPath *path, const bool use_polling_completion) {
  // Setup the send request
  TakyonSendRequest send_request = { .sub_buffer_count = 0,
                                     .sub_buffers = NULL,
                                     .use_is_sent_notification = true,
                                     .use_polling_completion = use_polling_completion,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void recvSignal(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for data to arrive
  uint64_t bytes_received;
  bool timed_out;
  takyonIsRecved(path, recv_request, RECV_TIMEOUT_SECONDS, &timed_out, &bytes_received, NULL);
  if (timed_out)  { printf("Timed out waiting for a message\n"); exit(EXIT_FAILURE); }
  assert(bytes_received == 0);

  // If the provider supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->capabilities.PostRecvs_supported) takyonPostRecvs(path, 1, recv_request);
}

static void twoSideThroughput(const bool is_endpointA, const char *provider, const uint64_t iterations, const uint64_t message_bytes, const uint32_t max_recv_requests, const bool use_polling_completion, const bool validate, const bool is_multi_threaded, TakyonBuffer *buffers) {
  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                            = is_endpointA;
  attrs.failure_mode                            = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                               = TAKYON_VERBOSITY_ERRORS;
  attrs.buffer_count                            = NUM_TAKYON_BUFFERS;
  attrs.buffers                                 = buffers;
  attrs.max_pending_send_and_one_sided_requests = 1;
  attrs.max_pending_recv_requests               = is_endpointA ? 1 : max_recv_requests;
  attrs.max_sub_buffers_per_send_request        = is_endpointA ? 1 : 0;  // 0 means zero-byte message
  attrs.max_sub_buffers_per_recv_request        = is_endpointA ? 0 : 1;  // 0 means zero-byte message

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recieves posted before sending can start
  uint32_t recv_request_count = is_endpointA ? 0 : max_recv_requests;
  TakyonSubBuffer *recver_sub_buffers = NULL;
  TakyonRecvRequest *recv_requests = NULL;
  TakyonRecvRequest repost_recv_request;
  if (is_endpointA) {
    // Only need a single zero-byte recv request to handle the re-post signaling
    recv_request_count = 1;
    recv_requests = &repost_recv_request;
    repost_recv_request.sub_buffer_count = 0;
    repost_recv_request.sub_buffers = NULL;
    repost_recv_request.use_polling_completion = use_polling_completion;
    repost_recv_request.usec_sleep_between_poll_attempts = 0;
  } else {
    // Setup all the message recvs
    recver_sub_buffers = calloc(max_recv_requests, sizeof(TakyonSubBuffer));
    recv_requests = calloc(max_recv_requests, sizeof(TakyonRecvRequest));
    for (uint32_t i=0; i<max_recv_requests; i++) {
      recver_sub_buffers[i].buffer_index = 1;
      recver_sub_buffers[i].bytes = message_bytes;
      recver_sub_buffers[i].offset = i*message_bytes;
      recv_requests[i].sub_buffer_count = 1;
      recv_requests[i].sub_buffers = &recver_sub_buffers[i];
      recv_requests[i].use_polling_completion = use_polling_completion;
      recv_requests[i].usec_sleep_between_poll_attempts = 0;
    }
  }

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, recv_request_count, recv_requests, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers, and calculate the throughput
  uint32_t recv_request_index = 0;
  double start_time = clockTimeSeconds();
  int64_t bytes_transferred = 0;
  double last_print_time = start_time - 1.0;
  uint64_t messages_transferred = 0;
  for (uint64_t i=0; i<iterations; i++) {
    // Transfer message
    if (path->attrs.is_endpointA) {
      // Send message
      sendMessage(path, message_bytes, use_polling_completion, validate, i+1);
    } else {
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      recvMessage(path, &recv_requests[recv_request_index], validate, i+1);
      recv_request_index = (recv_request_index + 1) % max_recv_requests;
    }

    // See if time to repost recvs (this is more efficient then posting a single recv after a message is received
    messages_transferred++;
    if (messages_transferred == max_recv_requests) {
      // Used up all the posted recvs. Need to repost
      if (path->attrs.is_endpointA) {
        // Wait for the recvs to be posted
        if (path->capabilities.IsRecved_supported) recvSignal(path, &repost_recv_request);
      } else {
        // If the provider supports pre-posting, then do it
        if (path->capabilities.PostRecvs_supported) takyonPostRecvs(path, max_recv_requests, recv_requests);
        // Let the send know the recvs are posted
        if (path->capabilities.Send_supported) sendSignal(path, use_polling_completion);
      }
      messages_transferred = 0;
    }

    // Print the current throughput
    double curr_time = clockTimeSeconds();
    double elapsed_time = curr_time - start_time;
    bytes_transferred += message_bytes;
    double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
    double Gb_per_sec = GB_per_sec * 8;
    double elapsed_print_time = curr_time - last_print_time;
    if (i == (iterations-1) || elapsed_print_time > 0.05) {
#ifdef ENABLE_CUDA
      const char *memory_type = "CUDA";
#else
      const char *memory_type = "CPU";
#endif
      if (path->attrs.is_endpointA) {
        if (!is_multi_threaded) {
          printf("\r%s (two-sided): " UINT64_FORMAT " %s transfers, %0.3f GB/sec, %0.3f Gb/sec", path->attrs.is_endpointA ? "Sender" : "Recver", i+1, memory_type, GB_per_sec, Gb_per_sec);
        }
      } else {
        printf("\r%s (two-sided): " UINT64_FORMAT " %s transfers, %0.3f GB/sec, %0.3f Gb/sec, dropped messages: " UINT64_FORMAT, path->attrs.is_endpointA ? "Sender" : "Recver", i+1, memory_type, GB_per_sec, Gb_per_sec, L_detected_drops);
      }
      fflush(stdout);
      last_print_time = curr_time;
    }
    if (elapsed_time >= 3.0) {
      start_time = curr_time;
      bytes_transferred = 0;
    }
  }
  if (!is_multi_threaded || !path->attrs.is_endpointA) {
    printf("\n");
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);
  if (!is_endpointA) {
    free(recver_sub_buffers);
    free(recv_requests);
  }
}

static void oneSideThroughput(const bool is_endpointA, const char *provider, const uint64_t iterations, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, TakyonBuffer *buffers) {
  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                            = is_endpointA;
  attrs.failure_mode                            = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                               = TAKYON_VERBOSITY_ERRORS;
  attrs.buffer_count                            = NUM_TAKYON_BUFFERS;
  attrs.buffers                                 = buffers;
  attrs.max_pending_send_and_one_sided_requests = 1;
  attrs.max_pending_recv_requests               = 0;
  attrs.max_sub_buffers_per_send_request        = 0;
  attrs.max_sub_buffers_per_recv_request        = 0;

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 0, NULL, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers, and calculate the throughput
  double start_time = clockTimeSeconds();
  int64_t bytes_transferred = 0;
  double last_print_time = start_time - 1.0;
  for (uint64_t i=0; i<iterations; i++) {
    // Transfer message
    if (path->attrs.is_endpointA) {
      // Send message
      writeMessage(path, message_bytes, use_polling_completion, validate, i+1);
      readMessage(path, message_bytes, use_polling_completion, validate, i+1);
    } else {
      // Enpoint B is not involved
    }

    // Print the current throughput
    if (path->attrs.is_endpointA) {
      double curr_time = clockTimeSeconds();
      double elapsed_time = curr_time - start_time;
      bytes_transferred += message_bytes*2;
      double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
      double Gb_per_sec = GB_per_sec * 8;
      double elapsed_print_time = curr_time - last_print_time;
      if (i == (iterations-1) || elapsed_print_time > 0.05) {
        printf("\rReader & Writer (one-sided): " UINT64_FORMAT " transfers, %0.3f GB/sec, %0.3f Gb/sec", i+1, GB_per_sec, Gb_per_sec);
        fflush(stdout);
        last_print_time = curr_time;
      }
      if (elapsed_time >= 3.0) {
        start_time = curr_time;
        bytes_transferred = 0;
      }
    }
  }
  if (path->attrs.is_endpointA) {
    printf("\n");
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);
}

void throughput(const bool is_endpointA, const char *provider, const uint64_t iterations, const uint64_t message_bytes, const uint32_t max_recv_requests, const bool use_polling_completion, const bool two_sided, const bool validate) {
  bool is_multi_threaded = (strncmp(provider, "InterThread ", 12) == 0);
  printf("Takyon Throughput: endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);
  if (!is_multi_threaded || is_endpointA) {
    printf("  Message Count:           " UINT64_FORMAT "\n", iterations);
    printf("  Message Bytes:           " UINT64_FORMAT "\n", message_bytes);
    printf("  Max Recv Requests:       %u\n", max_recv_requests);
    printf("  Completion Notification: %s\n", use_polling_completion ? "polling" : "event driven");
    printf("  Data Validation Enabled: %s\n", validate ? "yes" : "no");
  }
  if (validate && (message_bytes%8) != 0) { printf("When validation is enabled, message_bytes must be a multiple of 8\n"); exit(EXIT_FAILURE); }

  // Create the memory buffers used with transfering data
  // The 1st is for the sender, and the 2nd is for the receiver
  TakyonBuffer buffers[NUM_TAKYON_BUFFERS];
  for (uint32_t i=0; i<NUM_TAKYON_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
    if (two_sided) {
      buffer->bytes = (i==0) ? message_bytes : message_bytes * max_recv_requests;
    } else {
      buffer->bytes = (i==0) ? message_bytes*2 : message_bytes;
    }
    buffer->app_data = NULL;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer->addr, buffer->bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
    buffer->app_data = malloc(buffer->bytes); // Need a temp buffer for data validation
#else
#ifdef ENABLE_MMAP
    if (strncmp(provider, "InterProcess ", 13) == 0) {
      snprintf(buffer->name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_tp_buffer_%d_" UINT64_FORMAT, is_endpointA ? "A" : "B", i, buffer->bytes);
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

  // Create the path and do the transfers
  if (two_sided) {
    twoSideThroughput(is_endpointA, provider, iterations, message_bytes, max_recv_requests, use_polling_completion, validate, is_multi_threaded, buffers);
  } else {
    oneSideThroughput(is_endpointA, provider, iterations, message_bytes, use_polling_completion, validate, buffers);
  }

  // Free the takyon buffers
  for (uint32_t i=0; i<NUM_TAKYON_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
#ifdef ENABLE_CUDA
    cudaFree(buffer->addr);
    free(buffer->app_data); // Free temp buffer used for data validation
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

  printf("Takyon Throughput: %s is done.\n", is_endpointA ? "A" : "B");
}
