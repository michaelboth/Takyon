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

#define MAX_MESSAGE_BYTES 10000000
#define NUM_TAKYON_BUFFERS 2
#define MAX_RECV_REQUESTS 10 /*+ args */
#define USE_POLLING_COMPLETION false /*+ args */
/*+ validate data */

static void sendSignal(TakyonPath *path) {
  // Setup the send request
  TakyonSendRequest send_request = { .sub_buffer_count = 0,
                                     .sub_buffers = NULL,
                                     .use_is_sent_notification = true,
                                     .use_polling_completion = USE_POLLING_COMPLETION,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);

  // If the interconnect supports non blocking sends, then need to know when it's complete
  if (path->features.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void sendMessage(TakyonPath *path) {
  // Setup the send request
  TakyonSubBuffer sender_sub_buffer = { .buffer = &path->attrs.buffers[1], .bytes = MAX_MESSAGE_BYTES, .offset = 0 };
  TakyonSendRequest send_request = { .sub_buffer_count = 1,
                                     .sub_buffers = &sender_sub_buffer,
                                     .use_is_sent_notification = true, /*+ test without */
                                     .use_polling_completion = USE_POLLING_COMPLETION,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = 0;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);

  // If the interconnect supports non blocking sends, then need to know when it's complete
  if (path->features.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static void recvSignal(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for data to arrive
  uint64_t bytes_received;
  takyonIsRecved(path, recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, NULL);
  assert(bytes_received == 0);

  // If the interconnect supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->features.PostRecvs_supported) takyonPostRecvs(path, 1, recv_request);
}

static void recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for data to arrive
  uint64_t bytes_received;
  takyonIsRecved(path, recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, NULL);
  assert(bytes_received == recv_request->sub_buffers[0].bytes);

  // If the interconnect supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->features.PostRecvs_supported) takyonPostRecvs(path, 1, recv_request);
}

void throughput(const bool is_endpointA, const char *interconnect, const uint32_t iterations) {
  printf("Takyon Throughput (two-sided): endpoint %s: interconnect '%s'\n", is_endpointA ? "A" : "B", interconnect);

  // Create the memory buffers used with transfering data
  // The 1st is for the sender, and the 2nd is for the receiver
  TakyonBuffer buffers[NUM_TAKYON_BUFFERS];
  for (uint32_t i=0; i<NUM_TAKYON_BUFFERS; i++) {
    TakyonBuffer *buffer = &buffers[i];
    buffer->bytes = (i==0) ? MAX_MESSAGE_BYTES : MAX_MESSAGE_BYTES * MAX_RECV_REQUESTS;
    buffer->app_data = NULL;
#ifdef ENABLE_CUDA
    cudaError_t cuda_status = cudaMalloc(&buffer->addr, buffer->bytes);
    if (cuda_status != cudaSuccess) { printf("cudaMalloc() failed: %s\n", cudaGetErrorString(cuda_status)); exit(0); }
#else
#ifdef ENABLE_MMAP
    if (strncmp(interconnect, "InterProcess ", 13) == 0) {
      snprintf(buffer->name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_tp_buffer_%d_" UINT64_FORMAT, is_endpointA ? "A" : "B", i, buffer->bytes);
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
  attrs.buffer_count                            = NUM_TAKYON_BUFFERS;
  attrs.buffers                                 = buffers;
  attrs.max_pending_send_and_one_sided_requests = 1;
  attrs.max_pending_recv_requests               = is_endpointA ? 1 : MAX_RECV_REQUESTS;
  attrs.max_sub_buffers_per_send_request        = is_endpointA ? 1 : 0;  // 0 means zero-byte message
  attrs.max_sub_buffers_per_recv_request        = is_endpointA ? 0 : 1;  // 0 means zero-byte message

  /*+ GITHUB README: for each example, show list of Takyon features used */

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recieves posted before sending can start
  uint32_t recv_request_count = is_endpointA ? 0 : MAX_RECV_REQUESTS;
  TakyonSubBuffer *recver_sub_buffers = NULL;
  TakyonRecvRequest *recv_requests = NULL;
  TakyonRecvRequest repost_recv_request;
  if (recv_request_count > 0) {
    recver_sub_buffers = calloc(MAX_RECV_REQUESTS, sizeof(TakyonSubBuffer));
    recv_requests = calloc(MAX_RECV_REQUESTS, sizeof(TakyonRecvRequest));
    for (uint32_t i=0; i<MAX_RECV_REQUESTS; i++) {
      recver_sub_buffers[i].buffer = &buffers[1];
      recver_sub_buffers[i].bytes = MAX_MESSAGE_BYTES;
      recver_sub_buffers[i].offset = i*MAX_MESSAGE_BYTES;
      recv_requests[i].sub_buffer_count = 1;
      recv_requests[i].sub_buffers = &recver_sub_buffers[i];
      recv_requests[i].use_polling_completion = USE_POLLING_COMPLETION;
      recv_requests[i].usec_sleep_between_poll_attempts = 0;
    }
  } else {
    recv_request_count = 1;
    recv_requests = &repost_recv_request;
    repost_recv_request.sub_buffer_count = 0;
    repost_recv_request.sub_buffers = NULL;
    repost_recv_request.use_polling_completion = USE_POLLING_COMPLETION;
    repost_recv_request.usec_sleep_between_poll_attempts = 0;
  }

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, recv_request_count, recv_requests, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers, and calculate the throughput
  bool is_multi_threaded = (strncmp(interconnect, "InterThread ", 12) == 0);
  uint32_t recv_request_index = 0;
  double start_time = clockTimeSeconds();
  int64_t bytes_transferred = 0;
  double last_print_time = start_time - 1.0;
  for (uint32_t i=0; i<iterations; i++) {
    if (path->attrs.is_endpointA) {
      // Send message
      sendMessage(path);
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      if (path->features.IsRecved_supported) recvSignal(path, &repost_recv_request);
    } else {
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      recvMessage(path, &recv_requests[recv_request_index]);
      recv_request_index = (recv_request_index + 1) % MAX_RECV_REQUESTS;
      /*+ re-post in bulk */
      // Send a zero byte message to endpoint A to let it know it can send more messages
      if (path->features.Send_supported) sendSignal(path);
    }

    // Print the current throughput
    double curr_time = clockTimeSeconds();
    double elapsed_time = curr_time - start_time;
    bytes_transferred += MAX_MESSAGE_BYTES;
    double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
    double Gb_per_sec = GB_per_sec * 8;
    double elapsed_print_time = curr_time - last_print_time;
    if (elapsed_print_time > 0.05) {
      if (!is_multi_threaded || !path->attrs.is_endpointA) {
        printf("\r%s: %d transfers, %0.3f GB/sec, %0.3f Gb/sec", path->attrs.is_endpointA ? "Sender" : "Recver", i, GB_per_sec, Gb_per_sec);
        fflush(stdout);
      }
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
      if (!ok) { printf("mmapFree() failed: %s\n", error_message); exit(0); }
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
