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

#define FIRST_RECV_TIMEOUT_SECONDS 5.0 // Wait longer regardless if the connection is reliable or unreliable
#define ACTIVE_RECV_TIMEOUT_SECONDS 0.25

#ifdef ENABLE_CUDA
  static const char *MEMORY_TYPE = "CUDA";
#else
  static const char *MEMORY_TYPE = "CPU";
#endif

static uint32_t L_detected_drops = 0;
static uint32_t L_messages_recved = 0;

static void fillInMessage(TakyonPath *path, const uint64_t message_bytes, const uint64_t message_offset, const uint32_t message_count) {
#ifdef ENABLE_CUDA
  uint32_t *data_cpu = path->attrs.buffers[0].app_data;
  uint32_t *data_gpu = (uint32_t *)((uint64_t)path->attrs.buffers[0].addr + message_offset);
#else
  uint32_t *data_cpu = (uint32_t *)((uint64_t)path->attrs.buffers[0].addr + message_offset);
#endif
  uint64_t elements = message_bytes / sizeof(uint32_t);
  for (uint64_t i=0; i<elements; i++) {
    data_cpu[i] = (uint32_t)i + message_count;
  }
#ifdef ENABLE_CUDA
  cudaError_t cuda_status = cudaMemcpy(data_gpu, data_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#endif
}

static void validateMessage(TakyonPath *path, TakyonBuffer *buffer, uint64_t bytes_received, uint64_t message_bytes, const uint64_t message_offset, const uint32_t message_count, uint32_t *previous_start_value_inout) {
  // IMPORTANT: if recveiving from an RDMA UD (multicast or unicast) provider, then the receive buffer with start with the 40 byte RDMA Global Routing Header. Need to skip over this
  uint64_t rdma_grh_bytes = (strncmp(path->attrs.provider, "RdmaUD", 6) == 0) ? 40 : 0;
  bytes_received -= rdma_grh_bytes;
  message_bytes -= rdma_grh_bytes;

#ifdef ENABLE_CUDA
  uint32_t *data_cpu = buffer->app_data;
  uint32_t *data_gpu = (uint32_t *)((uint8_t *)buffer->addr + message_offset + rdma_grh_bytes);
  cudaError_t cuda_status = cudaMemcpy(data_cpu, data_gpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  uint32_t *data_cpu = (uint32_t *)((uint8_t *)buffer->addr + message_offset + rdma_grh_bytes);
#endif
  if (bytes_received != message_bytes) { printf("Message %u: Received " UINT64_FORMAT " bytes, but expect " UINT64_FORMAT " bytes\n", message_count, bytes_received, message_bytes); exit(EXIT_FAILURE); }
  if (message_bytes > 0) {
    uint32_t previous_start_value = *previous_start_value_inout;
    if (previous_start_value == data_cpu[0]) { printf("Message %u: Message start value=%u did not change from previous message value=%u. Is the sender also using '-v', or is this a duplicate multicast packet?\n", message_count, data_cpu[0], previous_start_value); exit(EXIT_FAILURE); }
    uint64_t elements = message_bytes / sizeof(uint32_t);
    for (uint64_t i=1; i<elements; i++) {
      if ((data_cpu[i-1]+1) != data_cpu[i]) { printf("Message %u: data[" UINT64_FORMAT "]=%u and data[" UINT64_FORMAT "]=%u did not increase by 1\n", message_count, i-1, data_cpu[i-1], i, data_cpu[i]); exit(EXIT_FAILURE); }
    }
    // Count drops
    if (data_cpu[0] > previous_start_value) {
      L_detected_drops += data_cpu[0] - (previous_start_value+1);
    }
    *previous_start_value_inout = data_cpu[0];
  }
}

static void sendMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint32_t message_count) {
  uint32_t message_index = (message_count-1) % path->attrs.max_pending_send_and_one_sided_requests;
  uint64_t message_offset = message_index * message_bytes;
  if (validate && message_bytes > 0) {
    fillInMessage(path, message_bytes, message_offset, message_count);
  }

  // Setup the send request
  bool use_sent_notification = (message_count % path->attrs.max_pending_send_and_one_sided_requests) == 0; // Need to get sent notification before out of send_requests or else the provider will get a buffer overflow
  TakyonSubBuffer sender_sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = message_offset };
  TakyonSendRequest send_request = { .sub_buffer_count = (message_bytes==0) ? 0 : 1,
                                     .sub_buffers = (message_bytes==0) ? NULL : &sender_sub_buffer,
                                     .use_is_sent_notification = use_sent_notification,
                                     .use_polling_completion = use_polling_completion,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggy_back_message = message_count;
  takyonSend(path, &send_request, piggy_back_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static bool recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request, const bool validate, const uint32_t message_count, uint32_t iterations) {
  // Wait for data to arrive
  uint64_t bytes_received;
  bool timed_out;
  uint32_t piggy_back_message;
  double timeout = (message_count==1) ? FIRST_RECV_TIMEOUT_SECONDS : ACTIVE_RECV_TIMEOUT_SECONDS;
  takyonIsRecved(path, recv_request, timeout, &timed_out, &bytes_received, &piggy_back_message);
  if (timed_out)  {
    uint32_t detected_drops = iterations-L_messages_recved;
    double drop_percent = 100.0 * (detected_drops / (double)iterations);
    printf("\nTimed out waiting for messages: dropped %u of %u (%0.2f%%).\n", detected_drops, iterations, drop_percent);
    return false;
  }
  if (bytes_received != recv_request->sub_buffers[0].bytes) {
    if (strncmp(path->attrs.provider, "RdmaUD", 6) == 0) {
      printf("\nGot " UINT64_FORMAT " bytes but expected " UINT64_FORMAT ". Make sure the sender matches byte size (RDMA UD receiver needs 40 extra bytes)\n", bytes_received-40, recv_request->sub_buffers[0].bytes-40);
    } else {
      printf("\nGot " UINT64_FORMAT " bytes but expected " UINT64_FORMAT ". Make sure the sender matches byte size\n", bytes_received, recv_request->sub_buffers[0].bytes);
    }
    exit(EXIT_FAILURE);
  }
  L_messages_recved++;

  if (validate) {
    static uint32_t previous_start_value = 0;
    uint64_t message_bytes = recv_request->sub_buffers[0].bytes;
    uint64_t message_offset = recv_request->sub_buffers[0].offset;
    TakyonBuffer *buffer = &path->attrs.buffers[recv_request->sub_buffers[0].buffer_index];
    validateMessage(path, buffer, bytes_received, message_bytes, message_offset, message_count, &previous_start_value);
  } else if (path->capabilities.piggy_back_messages_supported) {
    // Drop detection if validation is turned off
    static uint32_t previous_start_value = 0;
    if (previous_start_value == piggy_back_message) { printf("Message %u: Piggy back message=%u did not change from previous message. Is this a duplicate multicast packet?\n", message_count, piggy_back_message); exit(EXIT_FAILURE); }
    L_detected_drops += piggy_back_message - (previous_start_value+1);
    previous_start_value = piggy_back_message;
  }

  return true;
}

static void writeMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint32_t message_count) {
  uint32_t message_index = (message_count-1) % path->attrs.max_pending_send_and_one_sided_requests;
  uint64_t message_offset = message_index * message_bytes;
  if (validate) {
    fillInMessage(path, message_bytes, message_offset, message_count);
  }

  // Setup the one-sided write request
  bool use_done_notification = (message_count % path->attrs.max_pending_send_and_one_sided_requests) == 0; // Need to get done notification before all internal transfer buffers are used up
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = message_offset };
  TakyonOneSidedRequest request = { .is_write_request = true,
                                    .sub_buffer_count = 1,
                                    .sub_buffers = &sub_buffer,
                                    .remote_buffer_index = 0,
                                    .remote_offset = message_offset,
                                    .use_is_done_notification = use_done_notification,
                                    .use_polling_completion = use_polling_completion,
                                    .usec_sleep_between_poll_attempts = 0 };

  // Start the write
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported && request.use_is_done_notification) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);
}

static void readMessage(TakyonPath *path, const uint64_t message_bytes, const bool use_polling_completion, const bool validate, const uint32_t message_count) {
  // Setup the one-sided write request
  uint32_t message_index = (message_count-1) % path->attrs.max_pending_send_and_one_sided_requests;
  uint64_t message_offset = message_index * message_bytes;
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = path->attrs.max_pending_send_and_one_sided_requests * message_bytes + message_offset };
  TakyonOneSidedRequest request = { .is_write_request = false,
                                    .sub_buffer_count = 1,
                                    .sub_buffers = &sub_buffer,
                                    .remote_buffer_index = 0,
                                    .remote_offset = message_offset,
                                    .use_is_done_notification = true,
                                    .use_polling_completion = use_polling_completion,
                                    .usec_sleep_between_poll_attempts = 0 };

  // Start the write
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_supported) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);

  if (validate) {
    static uint32_t previous_start_value = 0;
    TakyonBuffer *buffer = &path->attrs.buffers[0];
    validateMessage(path, buffer, message_bytes, message_bytes, message_offset, message_count, &previous_start_value);
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

static bool recvSignal(TakyonPath *path, TakyonRecvRequest *recv_request) {
  // Wait for data to arrive
  uint64_t bytes_received;
  bool timed_out;
  takyonIsRecved(path, recv_request, ACTIVE_RECV_TIMEOUT_SECONDS, &timed_out, &bytes_received, NULL);
  if (timed_out) {
    printf("\nTimed out waiting for signal. Make sure both endpoints define the same number of recv buffers\n");
    return false;
  }
  if (bytes_received != 0) { printf("\nExpected a zero-byte message, but got " UINT64_FORMAT " bytes.\n", bytes_received); exit(EXIT_FAILURE); }

  // If the provider supports pre-posting, then need to post the recv to be ready for the next send, before the send starts
  if (path->capabilities.PostRecvs_supported) takyonPostRecvs(path, 1, recv_request);

  return true;
}

static void twoSidedThroughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t send_buffer_count, const uint32_t recv_buffer_count, const bool use_polling_completion, const bool validate, const bool is_multi_threaded, TakyonBuffer *buffer) {
  if ((recv_buffer_count%2) != 0) { printf("recv_buffer_count must be an even number to allows for re-posting after half the recv requests are used\n"); exit(EXIT_FAILURE); }

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                                   = is_endpointA;
  attrs.failure_mode                                   = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                                      = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                                   = (message_bytes==0) ? 0 : 1;
  attrs.buffers                                        = (message_bytes==0) ? NULL : buffer;
  attrs.max_pending_send_and_one_sided_requests        = is_endpointA ? send_buffer_count : 1;
  attrs.max_pending_recv_requests                      = is_endpointA ? 1 : recv_buffer_count;
  attrs.max_sub_buffers_per_send_and_one_sided_request = is_endpointA ? 1 : 0;  // 0 means zero-byte message
  attrs.max_sub_buffers_per_recv_request               = is_endpointA ? 0 : 1;  // 0 means zero-byte message

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recvs posted before sending can start
  uint32_t recv_request_count = is_endpointA ? 0 : recv_buffer_count;
  TakyonSubBuffer *recv_sub_buffers = NULL;
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
    // Setup all the recv requests, so they can be pre-posted at init
    recv_sub_buffers = calloc(recv_buffer_count, sizeof(TakyonSubBuffer));
    recv_requests = calloc(recv_buffer_count, sizeof(TakyonRecvRequest));
    for (uint32_t i=0; i<recv_buffer_count; i++) {
      recv_sub_buffers[i].buffer_index = 0;
      recv_sub_buffers[i].bytes = message_bytes;
      recv_sub_buffers[i].offset = i*message_bytes;
      recv_requests[i].sub_buffer_count = (message_bytes==0) ? 0 : 1;
      recv_requests[i].sub_buffers = (message_bytes==0) ? NULL : &recv_sub_buffers[i];
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
  uint32_t half_recv_buffer_count = recv_buffer_count/2;
  bool post_first_half = true;
  for (uint32_t i=0; i<iterations; i++) {
    // Transfer message
    if (path->attrs.is_endpointA) {
      // Send message
      sendMessage(path, message_bytes, use_polling_completion, validate, i+1);
    } else {
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      bool ok = recvMessage(path, &recv_requests[recv_request_index], validate, i+1, iterations);
      if (!ok) break; // Probably dropped packets and sender is done
      recv_request_index = (recv_request_index + 1) % recv_buffer_count;
    }

    // See if time to repost recvs (this is more efficient then posting a single recv after a message is received
    messages_transferred++;
    if (messages_transferred == half_recv_buffer_count) {
      // Used up all the posted recvs. Need to repost
      if (path->attrs.is_endpointA) {
        // Wait for the recvs to be posted, but if the provider is unreliable then no needed since dropped messages are allowed
        if (!path->capabilities.is_unreliable) {
          bool ok = recvSignal(path, &repost_recv_request);
          if (!ok) break; // Probably dropped packets and sender is done
        }
      } else {
        // If the provider supports pre-posting, then do it
	// IMPORTANT: Posting half at a time allows for data to continue arriving (in the old posted recvs) while new recvs are being posted
	TakyonRecvRequest *half_recv_requests = post_first_half ? recv_requests : &recv_requests[half_recv_buffer_count];
	post_first_half = !post_first_half;
        if (path->capabilities.PostRecvs_supported) takyonPostRecvs(path, half_recv_buffer_count, half_recv_requests);
        // Let the send know the recvs are posted, but if the provider is unreliable then no needed since dropped messages are allowed
        if (!path->capabilities.is_unreliable) sendSignal(path, use_polling_completion);
      }
      messages_transferred = 0;
    }

    // Print the current throughput
    double curr_time = clockTimeSeconds();
    double elapsed_time = curr_time - start_time;
    bytes_transferred += (message_bytes > 0) ? message_bytes : 4/*piggy back message*/;
    double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
    double Gb_per_sec = GB_per_sec * 8;
    double elapsed_print_time = curr_time - last_print_time;
    if (i == (iterations-1) || elapsed_print_time > 0.05) {
      if (path->attrs.is_endpointA) {
        if (!is_multi_threaded) {
          printf("\rSender (two-sided): sent %u %s messages, %0.3f GB/sec, %0.3f Gb/sec", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec);
        }
      } else {
	if (validate || path->capabilities.piggy_back_messages_supported) {
	  double drop_percent = 100.0 * (L_detected_drops / (double)iterations);
	  printf("\rRecver (two-sided): recved %u %s messages, %0.3f GB/sec, %0.3f Gb/sec, dropped messages: %u (%0.2f%%)", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec, L_detected_drops, drop_percent);
	} else {
	  printf("\rRecver (two-sided): recved %u %s messages, %0.3f GB/sec, %0.3f Gb/sec", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec);
	}
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
    free(recv_sub_buffers);
    free(recv_requests);
  }
}

static void oneSidedThroughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t send_buffer_count, const bool use_polling_completion, const bool validate, TakyonBuffer *buffer) {
  if (message_bytes == 0) { printf("Message bytes can't be zero for one-sided transfers\n"); exit(EXIT_FAILURE); }

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                                   = is_endpointA;
  attrs.failure_mode                                   = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                                      = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                                   = 1;
  attrs.buffers                                        = buffer;
  attrs.max_pending_send_and_one_sided_requests        = send_buffer_count;
  attrs.max_pending_recv_requests                      = 0;
  attrs.max_sub_buffers_per_send_and_one_sided_request = 1;
  attrs.max_sub_buffers_per_recv_request               = 0;

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 0, NULL, TAKYON_WAIT_FOREVER, &path);

  // Do the transfers, and calculate the throughput
  double start_time = clockTimeSeconds();
  int64_t bytes_transferred = 0;
  double last_print_time = start_time - 1.0;
  for (uint32_t i=0; i<iterations; i++) {
    // Transfer message
    if (path->attrs.is_endpointA) {
      // Send message
      writeMessage(path, message_bytes, use_polling_completion, validate, i+1);
      readMessage(path, message_bytes, use_polling_completion, validate, i+1);
    } else {
      // Endpoint B is not involved
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
        printf("\rReader & Writer (one-sided): completed %u %s transfers, %0.3f GB/sec, %0.3f Gb/sec", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec);
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

void throughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t send_buffer_count, const uint32_t recv_buffer_count, const bool use_polling_completion, const bool two_sided, const bool validate) {
  // Print greeting
  bool is_multi_threaded = (strncmp(provider, "InterThread", 11) == 0);
  printf("Takyon Throughput: endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);
  if (!is_multi_threaded || is_endpointA) {
    printf("  Message Transfer Count:       %u\n", iterations);
    printf("  Message Bytes:                " UINT64_FORMAT "\n", message_bytes);
    printf("  Send/Read/Write Buffer Count: %u\n", send_buffer_count);
    printf("  Recv Buffer Count:            %u\n", recv_buffer_count);
    printf("  Completion Notification:      %s\n", use_polling_completion ? "polling" : "event driven");
    printf("  Data Validation Enabled:      %s\n", validate ? "yes" : "no");
  }
  if (iterations > 0xfffffff) { printf("Message count needs to be <= %u\n", 0xfffffff); exit(EXIT_FAILURE); }
  if (validate && (message_bytes%4) != 0) { printf("When validation is enabled, message_bytes must be a multiple of 4\n"); exit(EXIT_FAILURE); }

  // Create the memory buffer used with transfering data
  // Only need one allocation that is then split with sub buffers
  TakyonBuffer buffer;
  if (message_bytes > 0) {
    if (two_sided) {
      buffer.bytes = (is_endpointA) ? message_bytes*send_buffer_count : message_bytes*recv_buffer_count;
    } else {
      buffer.bytes = (is_endpointA) ? 2*message_bytes*send_buffer_count : message_bytes*send_buffer_count;
    }
    buffer.app_data = NULL;
#ifdef ENABLE_CUDA
    buffer.app_data = calloc(1, buffer.bytes); // Need a temp buffer for data validation
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
      snprintf(buffer.name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s_throughput_buffer_" UINT64_FORMAT, is_endpointA ? "A" : "B", buffer.bytes);
#define MAX_ERROR_TEXT_BYTES 300
      char error_message[MAX_ERROR_TEXT_BYTES];
      bool ok = mmapAlloc(buffer.name, buffer.bytes, &buffer.addr, &buffer.app_data, error_message, MAX_ERROR_TEXT_BYTES);
      if (!ok) { printf("mmapAlloc() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      buffer.addr = calloc(1, buffer.bytes);
    }
#else
    buffer.addr = calloc(1, buffer.bytes);
#endif
#endif
  }

  // Create the path's endpoint and do the transfers
  if (two_sided) {
    twoSidedThroughput(is_endpointA, provider, iterations, message_bytes, send_buffer_count, recv_buffer_count, use_polling_completion, validate, is_multi_threaded, &buffer);
  } else {
    oneSidedThroughput(is_endpointA, provider, iterations, message_bytes, send_buffer_count, use_polling_completion, validate, &buffer);
  }

  // Free the takyon buffers
  if (message_bytes > 0) {
#ifdef ENABLE_CUDA
    cudaFree(buffer.addr);
    free(buffer.app_data); // Free temp buffer used for data validation
#else
#ifdef ENABLE_MMAP
    if (buffer.app_data != NULL) {
      char error_message[MAX_ERROR_TEXT_BYTES];
      bool ok = mmapFree(buffer.app_data, error_message, MAX_ERROR_TEXT_BYTES);
      if (!ok) { printf("mmapFree() failed: %s\n", error_message); exit(EXIT_FAILURE); }
    } else {
      free(buffer.addr);
    }
#else
    free(buffer.addr);
#endif
#endif
  }

  printf("Takyon Throughput: %s is done.\n", is_endpointA ? "A" : "B");
}
