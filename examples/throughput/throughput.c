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

static void fillInMessage(TakyonBuffer *buffer, const uint64_t message_bytes, const uint64_t message_offset, const uint32_t message_index) {
#ifdef ENABLE_CUDA
  uint32_t *data_cpu = buffer->app_data;
  uint32_t *data_gpu = (uint32_t *)((uint64_t)buffer->addr + message_offset);
#else
  uint32_t *data_cpu = (uint32_t *)((uint64_t)buffer->addr + message_offset);
#endif
  uint64_t elements = message_bytes / sizeof(uint32_t);
  for (uint64_t i=0; i<elements; i++) {
    data_cpu[i] = (uint32_t)i + (message_index+1);
  }
#ifdef ENABLE_CUDA
  cudaError_t cuda_status = cudaMemcpy(data_gpu, data_cpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#endif
}

static uint32_t validateMessage(TakyonBuffer *buffer, uint64_t message_bytes, const uint64_t message_offset, const uint32_t message_index, uint32_t *previous_start_value_inout) {
  // Get the addr to data
#ifdef ENABLE_CUDA
  uint32_t *data_cpu = buffer->app_data;
  uint32_t *data_gpu = (uint32_t *)((uint8_t *)buffer->addr + message_offset);
  cudaError_t cuda_status = cudaMemcpy(data_cpu, data_gpu, message_bytes, cudaMemcpyDefault);
  if (cuda_status != cudaSuccess) { printf("cudaMemcpy() failed: %s\n", cudaGetErrorString(cuda_status)); exit(EXIT_FAILURE); }
#else
  uint32_t *data_cpu = (uint32_t *)((uint8_t *)buffer->addr + message_offset);
#endif

  // See if the start value of data changed from the previous message
  uint32_t previous_start_value = *previous_start_value_inout;
  if (previous_start_value == data_cpu[0]) { printf("Message %u: Message start value=%u did not change from previous message value=%u. Is the sender also using '-v', or is this a duplicate multicast packet?\n", message_index+1, data_cpu[0], previous_start_value); exit(EXIT_FAILURE); }

  // Verify each value of the data increases by 1
  uint64_t elements = message_bytes / sizeof(uint32_t);
  for (uint64_t i=1; i<elements; i++) {
    if ((data_cpu[i-1]+1) != data_cpu[i]) { printf("Message %u: data[" UINT64_FORMAT "]=%u and data[" UINT64_FORMAT "]=%u did not increase by 1\n", message_index+1, i-1, data_cpu[i-1], i, data_cpu[i]); exit(EXIT_FAILURE); }
  }

  // Count dropped messages, but only if the start value increase; if start value descrease, may mean the sender restarted (can only viable with multicast and unicast)
  uint32_t detected_drops = 0;
  if (data_cpu[0] > previous_start_value) {
    detected_drops = data_cpu[0] - (previous_start_value+1);
  }

  *previous_start_value_inout = data_cpu[0];

  return detected_drops;
}

static void sendMessage(TakyonPath *path, const uint64_t message_bytes, uint64_t message_offset, const bool use_polling_completion, const uint32_t message_index) {
  // Setup the send request
  bool use_sent_notification = ((message_index+1) % path->attrs.max_pending_send_requests) == 0; // Need to get sent notification to implicitly wait for all pending transfers to complete or else the provider may get a buffer overflow
  TakyonSubBuffer sender_sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = message_offset };
  TakyonSendRequest send_request = { .sub_buffer_count = (message_bytes==0) ? 0 : 1,
                                     .sub_buffers = (message_bytes==0) ? NULL : &sender_sub_buffer,
                                     .submit_fence = false,
                                     .use_is_sent_notification = use_sent_notification,
                                     .use_polling_completion = use_polling_completion,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggyback_message = message_index+1;
  takyonSend(path, &send_request, piggyback_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_function_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
}

static bool recvMessage(TakyonPath *path, TakyonRecvRequest *recv_request, const uint32_t message_index, uint32_t iterations, uint32_t messages_received, uint64_t *bytes_received_out, uint32_t *piggyback_message_out) {
  // Wait for data to arrive
  uint64_t bytes_received;
  bool timed_out;
  uint32_t piggyback_message;
  double timeout = (message_index==0) ? FIRST_RECV_TIMEOUT_SECONDS : ACTIVE_RECV_TIMEOUT_SECONDS;
  takyonIsRecved(path, recv_request, timeout, &timed_out, &bytes_received, &piggyback_message);

  if (timed_out)  {
    uint32_t detected_drops = iterations-messages_received;
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

  *bytes_received_out = bytes_received;
  *piggyback_message_out = piggyback_message;

  return true;
}

static void writeMessage(TakyonPath *path, const uint64_t message_bytes, const uint64_t message_offset, const bool use_polling_completion) {
  // Setup the one-sided write request
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = message_offset };
  TakyonOneSidedRequest request = { .operation = TAKYON_OP_WRITE,
                                    .sub_buffer_count = 1,
                                    .sub_buffers = &sub_buffer,
                                    .remote_buffer_index = 0,
                                    .remote_offset = message_offset,
                                    .submit_fence = false,
                                    .use_is_done_notification = false,
                                    .use_polling_completion = use_polling_completion,
                                    .usec_sleep_between_poll_attempts = 0 };

  // Start the 'write'
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_function_supported && request.use_is_done_notification) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);
}

static void readMessage(TakyonPath *path, const uint64_t message_bytes, const uint64_t message_offset, const bool use_polling_completion) {
  // Setup the one-sided write request
  TakyonSubBuffer sub_buffer = { .buffer_index = 0, .bytes = message_bytes, .offset = message_offset };
  TakyonOneSidedRequest request = { .operation = TAKYON_OP_READ,
                                    .sub_buffer_count = 1,
                                    .sub_buffers = &sub_buffer,
                                    .remote_buffer_index = 0,
                                    .remote_offset = message_offset,
                                    .submit_fence = false,
                                    .use_is_done_notification = true,
                                    .use_polling_completion = use_polling_completion,
                                    .usec_sleep_between_poll_attempts = 0 };

  // Start the 'read'
  takyonOneSided(path, &request, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsOneSidedDone_function_supported && request.use_is_done_notification) takyonIsOneSidedDone(path, &request, TAKYON_WAIT_FOREVER, NULL);
}

static void sendSignal(TakyonPath *path, const bool use_polling_completion) {
  // Setup the send request
  TakyonSendRequest send_request = { .sub_buffer_count = 0,
                                     .sub_buffers = NULL,
                                     .submit_fence = false,
                                     .use_is_sent_notification = true,
                                     .use_polling_completion = use_polling_completion,
                                     .usec_sleep_between_poll_attempts = 0 };

  // Start the send
  uint32_t piggyback_message = 0;
  takyonSend(path, &send_request, piggyback_message, TAKYON_WAIT_FOREVER, NULL);

  // If the provider supports non blocking sends, then need to know when it's complete
  if (path->capabilities.IsSent_function_supported && send_request.use_is_sent_notification) takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
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
  if (path->capabilities.PostRecvs_function_supported) takyonPostRecvs(path, 1, recv_request);

  return true;
}

static void twoSidedThroughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t src_buffer_count, const uint32_t dest_buffer_count, const bool use_polling_completion, const bool validate, const bool is_multi_threaded, TakyonBuffer *buffer) {
  if ((dest_buffer_count%2) != 0) { printf("-dbufs=<n> must be an even number to allow for re-posting after half the recv requests are used\n"); exit(EXIT_FAILURE); }

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                      = is_endpointA;
  attrs.failure_mode                      = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                         = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                      = (message_bytes==0) ? 0 : 1;
  attrs.buffers                           = (message_bytes==0) ? NULL : buffer;
  attrs.max_pending_send_requests         = is_endpointA ? src_buffer_count : 1;
  attrs.max_pending_recv_requests         = is_endpointA ? 1 : dest_buffer_count;
  attrs.max_pending_write_requests        = 0;
  attrs.max_pending_read_requests         = 0;
  attrs.max_pending_atomic_requests       = 0;
  attrs.max_sub_buffers_per_send_request  = is_endpointA ? 1 : 0;  // 0 means zero-byte message
  attrs.max_sub_buffers_per_recv_request  = is_endpointA ? 0 : 1;  // 0 means zero-byte message
  attrs.max_sub_buffers_per_write_request = 0;
  attrs.max_sub_buffers_per_read_request  = 0;

  // Setup the receive request and it's sub buffer
  //   - This is done before the path is setup in the case the receiver needs the recvs posted before sending can start
  uint32_t recv_request_count = is_endpointA ? 0 : dest_buffer_count;
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
    recv_sub_buffers = calloc(dest_buffer_count, sizeof(TakyonSubBuffer));
    recv_requests = calloc(dest_buffer_count, sizeof(TakyonRecvRequest));
    for (uint32_t i=0; i<dest_buffer_count; i++) {
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
  uint32_t messages_received = 0;
  uint32_t messages_to_be_reposted = 0;
  uint32_t detected_drops = 0;
  uint32_t half_dest_buffer_count = dest_buffer_count/2;
  bool post_first_half = true;
  bool provider_is_RdmaUD = (strncmp(path->attrs.provider, "RdmaUD", 6) == 0);
  for (uint32_t i=0; i<iterations; i++) {
    // Transfer message
    if (path->attrs.is_endpointA) {
      // Prepare message if validating
      uint32_t message_index = i % path->attrs.max_pending_send_requests;
      uint64_t message_offset = message_index * message_bytes;
      if (validate && message_bytes > 0) {
	TakyonBuffer *buffer = &path->attrs.buffers[0];
	fillInMessage(buffer, message_bytes, message_offset, i);
      }
      // Send message
      sendMessage(path, message_bytes, message_offset, use_polling_completion, i);

    } else {
      // Wait for the message to arrive (will reuse the recv_request that was already prepared)
      uint64_t bytes_received;
      uint32_t piggyback_message;
      TakyonRecvRequest *recv_request = &recv_requests[recv_request_index];
      bool ok = recvMessage(path, recv_request, i, iterations, messages_received, &bytes_received, &piggyback_message);
      if (!ok) break; // Probably dropped packets and sender is done
      messages_received++;

      // Validate message
      // Verify bytes received
      uint64_t expected_message_bytes = recv_request->sub_buffers[0].bytes;
      if (bytes_received != expected_message_bytes) { printf("Message %u: Received " UINT64_FORMAT " bytes, but expect " UINT64_FORMAT " bytes\n", i+1, bytes_received, expected_message_bytes); exit(EXIT_FAILURE); }
      uint64_t message_offset = recv_request->sub_buffers[0].offset;
      // IMPORTANT: if receiving from an RDMA UD (multicast or unicast) provider, then the receive buffer with start with the 40 byte RDMA Global Routing Header. Need to skip over this
      if (provider_is_RdmaUD) {
	bytes_received -= 40;
	message_offset += 40;
      }
      // Verify message data
      if (validate && bytes_received > 0) {
	static uint32_t previous_start_value = 0;
	TakyonBuffer *buffer = &path->attrs.buffers[recv_request->sub_buffers[0].buffer_index];
	detected_drops += validateMessage(buffer, bytes_received, message_offset, i, &previous_start_value);
      } else if (path->capabilities.piggyback_messages_supported) {
	// Validation is turned of but can still do drop detection if the piggyback message is suported
	static uint32_t previous_start_value = 0;
	if (previous_start_value == piggyback_message) { printf("Message %u: Piggyback message=%u did not change from previous message. Is this a duplicate multicast packet?\n", i+1, piggyback_message); exit(EXIT_FAILURE); }
	detected_drops += piggyback_message - (previous_start_value+1);
	previous_start_value = piggyback_message;
      }
      // Prepare for next recv request
      recv_request_index = (recv_request_index + 1) % dest_buffer_count;
    }

    // See if time to repost recvs (this is more efficient then posting a single recv after a message is received
    messages_to_be_reposted++;
    if (messages_to_be_reposted == half_dest_buffer_count) {
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
	TakyonRecvRequest *half_recv_requests = post_first_half ? recv_requests : &recv_requests[half_dest_buffer_count];
	post_first_half = !post_first_half;
        if (path->capabilities.PostRecvs_function_supported) takyonPostRecvs(path, half_dest_buffer_count, half_recv_requests);
        // Let the send know the recvs are posted, but if the provider is unreliable then no needed since dropped messages are allowed
        if (!path->capabilities.is_unreliable) sendSignal(path, use_polling_completion);
      }
      messages_to_be_reposted = 0;
    }

    // Print the current throughput
    double curr_time = clockTimeSeconds();
    double elapsed_time = curr_time - start_time;
    bytes_transferred += (message_bytes > 0) ? message_bytes : 4/*piggyback message*/;
    double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
    double Gb_per_sec = GB_per_sec * 8;
    double elapsed_print_time = curr_time - last_print_time;
    if (i == (iterations-1) || elapsed_print_time > 0.05) {
      if (path->attrs.is_endpointA) {
        if (!is_multi_threaded) {
          printf("\rSender (two-sided send/recv): sent %u %s messages, %0.3f GB/sec, %0.3f Gb/sec", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec);
        }
      } else {
	if (validate || path->capabilities.piggyback_messages_supported) {
	  double drop_percent = 100.0 * (detected_drops / (double)iterations);
	  printf("\rRecver (two-sided send/recv): recved %u %s messages, %0.3f GB/sec, %0.3f Gb/sec, dropped messages: %u (%0.2f%%)", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec, detected_drops, drop_percent);
	} else {
	  printf("\rRecver (two-sided send/recv): recved %u %s messages, %0.3f GB/sec, %0.3f Gb/sec (can't do drop detection)", i+1, MEMORY_TYPE, GB_per_sec, Gb_per_sec);
	}
      }
      fflush(stdout);
      last_print_time = curr_time;
    }
    if (elapsed_time >= 3.0) {
      // Reset stats
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

static void oneSidedThroughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t src_buffer_count, const uint32_t dest_buffer_count, const bool use_polling_completion, const bool validate, const bool is_multi_threaded, TakyonBuffer *buffer, const char *transfer_mode) {
  if (message_bytes == 0) { printf("-bytes=<n> can't be zero for one-sided transfers\n"); exit(EXIT_FAILURE); }
  if (src_buffer_count != dest_buffer_count) { printf("For one-sided transfers, -sbufs=<n> and -dbufs=<n> must be the same\n"); exit(EXIT_FAILURE); }
  bool transfer_mode_is_read = (strcmp(transfer_mode, "read")==0);

  // Define the path attributes
  //   - Can't be changed after path creation
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, provider, TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                      = is_endpointA;
  attrs.failure_mode                      = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                         = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                      = 1;
  attrs.buffers                           = buffer;                  // One buffer holds enough room for 'src_buffer_count' messages
  attrs.max_pending_send_requests         = 2;
  attrs.max_pending_recv_requests         = 2;
  attrs.max_pending_write_requests        = transfer_mode_is_read ? 0 : src_buffer_count;
  attrs.max_pending_read_requests         = transfer_mode_is_read ? src_buffer_count : 0;
  attrs.max_pending_atomic_requests       = 0;
  attrs.max_sub_buffers_per_send_request  = 0;
  attrs.max_sub_buffers_per_recv_request  = 0;
  attrs.max_sub_buffers_per_write_request = 1;
  attrs.max_sub_buffers_per_read_request  = 1;

  // Recv request used for signaling
  TakyonRecvRequest recv_requests[2];
  for (uint32_t i=0; i<2; i++) {
    recv_requests[i].sub_buffer_count = 0;
    recv_requests[i].sub_buffers = NULL;
    recv_requests[i].use_polling_completion = use_polling_completion;
    recv_requests[i].usec_sleep_between_poll_attempts = 0;
  }

  // Create one side of the path
  //   - The other side will be created in a different thread/process
  TakyonPath *path;
  (void)takyonCreate(&attrs, 2, recv_requests, TAKYON_WAIT_FOREVER, &path);

  uint32_t completed_iterations = 0;
  double start_time = clockTimeSeconds();
  double last_print_time = start_time - 1.0;
  int64_t bytes_transferred = 0;
  while (completed_iterations < iterations) {
    if (transfer_mode_is_read) {
      // 'read' throughput
      for (uint32_t half_index=0; half_index<2; half_index++) {
        if (path->attrs.is_endpointA) {
          // Wait for permission to start filling in the next batch of data (no need to wait if this is the first round of transfers)
          if (completed_iterations > 0) {
            recvSignal(path, &recv_requests[half_index]);
          }
          // Fill in the message
          if (validate) {
            for (uint32_t i=0; i<src_buffer_count/2; i++) {
              uint32_t i2 = (half_index==0) ? i : src_buffer_count/2+i;
              uint32_t iteration = completed_iterations + i2;
              uint32_t message_index = i2 % src_buffer_count;
              uint64_t message_offset = message_index * message_bytes;
              TakyonBuffer *buffer = &path->attrs.buffers[0]; // Source message will be put in first half of the buffer
              fillInMessage(buffer, message_bytes, message_offset, iteration);
            }
          }
          // Let the remote endpoint know the set of messages are ready to be read
          sendSignal(path, use_polling_completion);
        } else {
          // Wait for signal to inform that messages can be read
          recvSignal(path, &recv_requests[half_index]);
          // Read messages
          for (uint32_t i=0; i<src_buffer_count/2; i++) {
            uint32_t i2 = (half_index==0) ? i : src_buffer_count/2+i;
            uint32_t message_index = i2 % src_buffer_count;
            uint64_t message_offset = message_index * message_bytes;
            TakyonBuffer *buffer = &path->attrs.buffers[0]; // Result is in second half of the buffer
            readMessage(path, message_bytes, message_offset, use_polling_completion);
            // Validate messages
            if (validate) {
              static uint32_t previous_start_value = 0;
              validateMessage(buffer, message_bytes, message_offset, message_index, &previous_start_value);
            }
          }
          // Send signal to inform the remote endpoint that more message can be written
          sendSignal(path, use_polling_completion);
        }
      }

    } else {
      // 'write' throughput
      // Transfer in one half at a time to allow for overlapping of 'writes' and receiving signals
      for (uint32_t half_index=0; half_index<2; half_index++) {
        if (path->attrs.is_endpointA) {
          // Wait for permission to write (no need to wait if this is the first round of transfers)
          if (completed_iterations > 0) {
            recvSignal(path, &recv_requests[half_index]);
          }
          for (uint32_t i=0; i<src_buffer_count/2; i++) {
            uint32_t i2 = (half_index==0) ? i : src_buffer_count/2+i;
            uint32_t iteration = completed_iterations + i2;
            uint32_t message_index = i2 % src_buffer_count;
            uint64_t message_offset = message_index * message_bytes;
            if (validate) {
              // Fill in the message
              TakyonBuffer *buffer = &path->attrs.buffers[0]; // Source message will be put in first half of the buffer
              fillInMessage(buffer, message_bytes, message_offset, iteration);
            }
            // Write the message
            writeMessage(path, message_bytes, message_offset, use_polling_completion);
          }
          // Let the remote endpoint know the set of messages arrived
          sendSignal(path, use_polling_completion);

        } else {
          // Wait for signal to inform that messages were written
          recvSignal(path, &recv_requests[half_index]);
          // Validate messages
          if (validate) {
            static uint32_t previous_start_value = 0;
            for (uint32_t i=0; i<src_buffer_count/2; i++) {
              uint32_t i2 = (half_index==0) ? i : src_buffer_count/2+i;
              uint32_t message_index = i2 % src_buffer_count;
              uint64_t message_offset = message_index * message_bytes;
              TakyonBuffer *buffer = &path->attrs.buffers[0]; // Result is in second half of the buffer
              validateMessage(buffer, message_bytes, message_offset, message_index, &previous_start_value);
            }
          }
          // Send signal to inform the remote endpoint that more message can be written
          sendSignal(path, use_polling_completion);
        }
      }
    }
    completed_iterations += src_buffer_count;

    // Print stats
    if (!is_multi_threaded || !path->attrs.is_endpointA) {
      double curr_time = clockTimeSeconds();
      double elapsed_time = curr_time - start_time;
      bytes_transferred += src_buffer_count * message_bytes;
      double GB_per_sec = (bytes_transferred / 1000000000.0) / elapsed_time;
      double Gb_per_sec = GB_per_sec * 8;
      double elapsed_print_time = curr_time - last_print_time;
      if (completed_iterations >= iterations || elapsed_print_time > 0.05) {
        printf("\r%s: completed %u %s '%s' transfers, %0.3f GB/sec, %0.3f Gb/sec", path->attrs.is_endpointA ? "A" : "B", completed_iterations, MEMORY_TYPE, transfer_mode_is_read ? "read" : "write", GB_per_sec, Gb_per_sec);
        fflush(stdout);
        last_print_time = curr_time;
      }
      if (elapsed_time >= 3.0) {
        // Reset stats
        start_time = curr_time;
        bytes_transferred = 0;
      }
    }
  }

  if (!is_multi_threaded || !path->attrs.is_endpointA) {
    printf("\n");
  }

  // Destroy the path
  takyonDestroy(path, TAKYON_WAIT_FOREVER);
}

void throughput(const bool is_endpointA, const char *provider, const uint32_t iterations, const uint64_t message_bytes, const uint32_t src_buffer_count, const uint32_t dest_buffer_count, const bool use_polling_completion, const char *transfer_mode, const bool validate) {
  // Print greeting
  bool is_multi_threaded = (strncmp(provider, "InterThread", 11) == 0);
  printf("Takyon Throughput: endpoint %s: provider '%s'\n", is_endpointA ? "A" : "B", provider);
  if (!is_multi_threaded || is_endpointA) {
    printf("  Message Transfer Iterations:  -i=%u\n", iterations);
    printf("  Message Bytes:            -bytes=" UINT64_FORMAT "\n", message_bytes);
    printf("  Source Buffer Count:      -sbufs=%u\n", src_buffer_count);
    printf("  Dest Buffer Count:        -dbufs=%u\n", dest_buffer_count);
    printf("  Completion Notification:         %s\n", use_polling_completion ? "polling" : "event driven (-e)");
    printf("  Data Validation Enabled:         %s\n", validate ? "yes (-V)" : "no");
  }
  if (iterations > 0xfffffff) { printf("-i=<n> needs to be <= %u\n", 0xfffffff); exit(EXIT_FAILURE); }
  if ((src_buffer_count%2) != 0) { printf("-sbufs=<n> must be an even number\n"); exit(EXIT_FAILURE); }
  if (validate && (message_bytes%4) != 0) { printf("When validation is enabled, -bytes=<n> must be a multiple of 4\n"); exit(EXIT_FAILURE); }

  // Create the memory buffer used with transfering data
  // Only need one allocation that is then split with sub buffers
  TakyonBuffer buffer;
  if (message_bytes > 0) {
    buffer.bytes = (is_endpointA) ? message_bytes*src_buffer_count : message_bytes*dest_buffer_count;
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
  if (strcmp(transfer_mode, "send/recv")==0) {
    twoSidedThroughput(is_endpointA, provider, iterations, message_bytes, src_buffer_count, dest_buffer_count, use_polling_completion, validate, is_multi_threaded, &buffer);
  } else {
    oneSidedThroughput(is_endpointA, provider, iterations, message_bytes, src_buffer_count, dest_buffer_count, use_polling_completion, validate, is_multi_threaded, &buffer, transfer_mode);
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
