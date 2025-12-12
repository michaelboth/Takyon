//     Copyright 2025 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

#include "ThroughputTest.hpp"
#include "takyon.h"
#include <cstring>

// For Unikorn instrumentation
#include "unikorn_instrumentation.h"
#include "unikorn_macros.h" /*+ merge into unikorn_instrumentation.h to avoid requireing Unikorn for customer */

static const uint64_t ACK_BYTES = sizeof(int);

// Algorithm:
//   - Send multiple messages (one per transport buffer) at once and use a round trip acknowledgement to synchronize the re-used of tranport buffers
//
// Algorithm:
//   loop {
//     if (is_sender) {
//       if (not_first_iteration) waitForAck and repost
//       for (each buffer) {
//         if (prev_send_request_in_use) waitForSendCompletion
//         send
//       }
//     } else {  // is_receiver
//       // First half
//       for (first half of buffers) {
//         recv
//       }
//       repostFirstHalfOfBuffers
//       if (prev_first_half_ACK_in_progress) waitForSendCompletion
//       sendFirstHalfAck
//
//       // Second half
//       for (second half of buffers) {
//         recv
//       }
//       repostSecondHalfOfBuffers
//       if (prev_second_half_ACK_in_progress) waitForSendCompletion
//       sendSecondHalfAck
//     }
//   }
//
//  For most accurate results use multiple buffers.
//  For best results use polling with no sleep delay, since event-driven completion requires a thread based context switch

static void waitForAck(TakyonPath *_path, TakyonRecvRequest *_recv_request, uint64_t _expected_bytes) {
  // Wait for ACK
  uint64_t bytes_received = 0;
  (void)takyonIsRecved(_path, _recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, NULL);
  if (bytes_received != _expected_bytes) { EXIT_WITH_MESSAGE(std::string("Received ACK bytes should be " + std::to_string(_expected_bytes) + " but got " + std::to_string(bytes_received))); }

  // Re-post the recv
  if (_path->capabilities.PostRecvs_function_supported) {
    uint32_t request_count = 1;
    (void)takyonPostRecvs(_path, request_count, _recv_request);
  }
}

void ThroughputTest::runThroughputTest(bool _is_sender, const Common::AppParams &_app_params) {
  UK_RECORD_EVENT(_app_params.unikorn_session, THROUGHPUT_TEST_START_ID, 0);

  // Allocate transport memory
  uint64_t total_send_bytes = _is_sender ? _app_params.nbufs * _app_params.nbytes : ACK_BYTES * 2;
  uint64_t extra_recv_bytes = (_app_params.provider == "UD") ? 40 : 0; // RDMA UD receiver needs to allocate 40 extra bytes for RDMA's global routing header
  uint64_t total_recv_bytes = _is_sender ? 2*(ACK_BYTES+extra_recv_bytes) : _app_params.nbufs * (_app_params.nbytes+extra_recv_bytes);
  bool is_for_rdma = (_app_params.provider == "RC" || _app_params.provider == "UC" || _app_params.provider == "UD");
  void *send_memory = Common::allocateTransportMemory(total_send_bytes, is_for_rdma);
  void *recv_memory = Common::allocateTransportMemory(total_recv_bytes, is_for_rdma);

  // Fill in some helpful send side data
  if (_is_sender) {
    /*+ do this in main loop and vary */
    uint64_t integer_count = total_send_bytes / sizeof(int);
    uint32_t *send_memory_integer = (uint32_t *)send_memory;
    for (uint64_t i=0; i<integer_count; i++) {
      send_memory_integer[i] = (uint32_t)i;
    }
  }

  // Create the Takyon transport buffers
  TakyonBuffer transport_buffers[2];
  // Send buffer
  {
    TakyonBuffer *buffer = &transport_buffers[0];
    buffer->bytes = total_send_bytes;
    buffer->app_data = NULL;
    buffer->addr = send_memory;
  }
  // Recv buffer
  {
    TakyonBuffer *buffer = &transport_buffers[1];
    buffer->bytes = total_recv_bytes;
    buffer->app_data = NULL;
    buffer->addr = recv_memory;
  }

  // Setup the recv requests
  uint32_t recv_request_count = _is_sender ? 2 : _app_params.nbufs;
  TakyonSubBuffer *recv_sub_buffers = new TakyonSubBuffer[recv_request_count];
  TakyonRecvRequest *recv_requests = new TakyonRecvRequest[recv_request_count];
  for (uint32_t i=0; i<recv_request_count; i++) {
    // Sub buffer
    recv_sub_buffers[i].buffer_index = 1; // Second transport buffer
    recv_sub_buffers[i].bytes = _is_sender ? ACK_BYTES + extra_recv_bytes : _app_params.nbytes + extra_recv_bytes;
    recv_sub_buffers[i].offset = _is_sender ? i * (ACK_BYTES + extra_recv_bytes) : i * (_app_params.nbytes + extra_recv_bytes);
    // Request
    recv_requests[i].sub_buffer_count = 1;
    recv_requests[i].sub_buffers = &recv_sub_buffers[i];
    recv_requests[i].use_polling_completion = _app_params.use_polling;
    recv_requests[i].usec_sleep_between_poll_attempts = 0;  // Use 0 since this test requires full speed completion monitoring to minimize throughput
    recv_requests[i].app_data = NULL;
  }

  // Setup the send requests
  uint32_t send_request_count = _is_sender ? _app_params.nbufs : 2;
  bool *send_request_in_use = new bool[send_request_count];
  TakyonSubBuffer *send_sub_buffers = new TakyonSubBuffer[send_request_count];
  TakyonSendRequest *send_requests = new TakyonSendRequest[send_request_count];
  for (uint32_t i=0; i<send_request_count; i++) {
    send_request_in_use[i] = false;
    // Sub buffer
    send_sub_buffers[i].buffer_index = 0; // First transport buffer
    send_sub_buffers[i].bytes = _is_sender ? _app_params.nbytes : ACK_BYTES;
    send_sub_buffers[i].offset = _is_sender ? i * _app_params.nbytes : i * ACK_BYTES;
    // Request
    send_requests[i].sub_buffer_count = 1;
    send_requests[i].sub_buffers = &send_sub_buffers[i];
    send_requests[i].submit_fence = false; // Not needed since not using read or atomics
    send_requests[i].use_is_sent_notification = true;
    send_requests[i].use_polling_completion = _app_params.use_polling;
    send_requests[i].usec_sleep_between_poll_attempts = 0;  // Use 0 since this test requires full speed completion monitoring to minimize throughput
    send_requests[i].app_data = NULL;
  }

  // Setup Takyon attributes
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, _app_params.provider_params.c_str(), TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                      = _is_sender;
  attrs.failure_mode                      = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                         = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                      = 2;
  attrs.buffers                           = transport_buffers;
  attrs.max_pending_send_requests         = send_request_count;
  attrs.max_pending_recv_requests         = recv_request_count;
  attrs.max_pending_write_requests        = 0; // Not used
  attrs.max_pending_read_requests         = 0; // Not used
  attrs.max_pending_atomic_requests       = 0; // Not used
  attrs.max_sub_buffers_per_send_request  = 1;
  attrs.max_sub_buffers_per_recv_request  = 1;
  attrs.max_sub_buffers_per_write_request = 0; // Not used
  attrs.max_sub_buffers_per_read_request  = 0; // Not used

  // Init
  if (_app_params.verbose) printf("Initializing throughput connection...\n");
  UK_RECORD_EVENT(_app_params.unikorn_session, INIT_START_ID, 0);
  uint32_t post_recv_count = recv_request_count;
  TakyonPath *path = NULL;
  (void)takyonCreate(&attrs, post_recv_count, recv_requests, TAKYON_WAIT_FOREVER, &path);
  UK_RECORD_EVENT(_app_params.unikorn_session, INIT_END_ID, 0);

  // Throught main loop
  if (_app_params.verbose) printf("Collecting throughput results...\n");
  /*+ if _app_params.iters == 0, then choose a good size based on nbytes */
  double start_time = Common::clockTimeSeconds();
  double print_start_time = -Common::ELAPSED_SECONDS_TO_PRINT; // Make sure first pass prints
  double accumulated_throughput_Gbps = -1.0;
  do {
    uint32_t iter = 0;
    while (iter<_app_params.iters) {
      for (uint32_t half_index=0; half_index<2; half_index++) {
        if (_is_sender) {
          // Wait for ACK if not the first iteration
          if (iter > 0) {
            UK_RECORD_EVENT(_app_params.unikorn_session, WAIT_FOR_ACK_START_ID, 0);
            waitForAck(path, &recv_requests[half_index], ACK_BYTES);
            UK_RECORD_EVENT(_app_params.unikorn_session, WAIT_FOR_ACK_END_ID, 0);
          }

          // Send messages (1 per buffer)
          UK_RECORD_EVENT(_app_params.unikorn_session, SEND_BUFFERS_START_ID, 0);
          for (uint32_t i=0; i<_app_params.nbufs/2; i++) {
            uint32_t send_index = i + half_index * _app_params.nbufs/2;
            TakyonSendRequest *send_request = &send_requests[send_index];
            // If this send request was used before, make sure to wait until it's done being used
            if (path->capabilities.IsSent_function_supported && send_request_in_use[send_index]) {
              (void)takyonIsSent(path, send_request, TAKYON_WAIT_FOREVER, NULL);
              send_request_in_use[send_index] = false;
            }
            send_request_in_use[send_index] = true;
            // Send the data
            uint32_t piggyback_message = 0; // Ignoring since UDP sockets can't use it
            (void)takyonSend(path, send_request, piggyback_message, TAKYON_WAIT_FOREVER, NULL);
          }
          UK_RECORD_EVENT(_app_params.unikorn_session, SEND_BUFFERS_END_ID, 0);

        } else {
          // Recv messages (1 per buffer)
          UK_RECORD_EVENT(_app_params.unikorn_session, RECV_BUFFERS_START_ID, 0);
          for (uint32_t i=0; i<_app_params.nbufs/2; i++) {
            uint32_t recv_index = i + half_index * _app_params.nbufs/2;
            // Wait for data
            TakyonRecvRequest *recv_request = &recv_requests[recv_index];
            TakyonSubBuffer *recv_sub_buffer = recv_request->sub_buffers;
            uint64_t bytes_received = 0;
            (void)takyonIsRecved(path, recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, NULL);
            if (bytes_received != recv_sub_buffer->bytes) { EXIT_WITH_MESSAGE(std::string("Received bytes should be " + std::to_string(recv_sub_buffer->bytes) + " but got " + std::to_string(bytes_received))); }
            if (_app_params.validate) {
              uint32_t *addr = (uint32_t *)((uint64_t)recv_memory + recv_sub_buffer->offset + extra_recv_bytes);
              uint64_t integer_offset = recv_sub_buffer->offset / sizeof(int);
              uint64_t integer_count = bytes_received / sizeof(int);
              for (uint64_t j=0; j<integer_count; j++) {
                uint64_t index = integer_offset + j;
                if (addr[j] != (uint32_t)index) {
                  EXIT_WITH_MESSAGE(std::string("Received invalid data, recv_index=" + std::to_string(recv_index) + ", j=" + std::to_string(j) + ", index=" + std::to_string(index) + " expected " + std::to_string((uint32_t)index) + " but got " + std::to_string(addr[j])));
                }
              }
            }
          }
          UK_RECORD_EVENT(_app_params.unikorn_session, RECV_BUFFERS_END_ID, 0);

          // Repost all the receives at once
          if (path->capabilities.PostRecvs_function_supported) {
            UK_RECORD_EVENT(_app_params.unikorn_session, POST_RECVS_START_ID, 0);
            uint32_t recv_index = half_index * _app_params.nbufs/2;
            TakyonRecvRequest *first_recv_request = &recv_requests[recv_index];
            (void)takyonPostRecvs(path, _app_params.nbufs/2, first_recv_request);
            UK_RECORD_EVENT(_app_params.unikorn_session, POST_RECVS_END_ID, 0);
          }

          // Send the ACK to the sender to get more data, and wait for the send to complete
          UK_RECORD_EVENT(_app_params.unikorn_session, SEND_ACK_START_ID, 0);
          // If this send request was used before, make sure to wait until it's done being used
          if (path->capabilities.IsSent_function_supported && send_request_in_use[half_index]) {
            (void)takyonIsSent(path, &send_requests[half_index], TAKYON_WAIT_FOREVER, NULL);
            send_request_in_use[half_index] = false;
          }
          uint32_t piggyback_message = 0; // Ignoring since UDP sockets can't use it
          (void)takyonSend(path, &send_requests[half_index], piggyback_message, TAKYON_WAIT_FOREVER, NULL);
          send_request_in_use[half_index] = true;
          UK_RECORD_EVENT(_app_params.unikorn_session, SEND_ACK_END_ID, 0);
        }
      }

      // Prepare for next iteration
      /*+ change byte size if enough time has passed */
      iter++;
    }

    // Wait for final ACKs
    if (_is_sender && _app_params.iters > 0) {
      UK_RECORD_EVENT(_app_params.unikorn_session, WAIT_FOR_ACK_START_ID, 0);
      waitForAck(path, &recv_requests[0], ACK_BYTES);
      waitForAck(path, &recv_requests[1], ACK_BYTES);
      UK_RECORD_EVENT(_app_params.unikorn_session, WAIT_FOR_ACK_END_ID, 0);
    }

    // Get throughput results
    double end_time = Common::clockTimeSeconds();
    double elapsed_seconds = end_time - start_time;
    start_time = end_time;
    uint64_t total_bytes = _app_params.iters * _app_params.nbufs * _app_params.nbytes;
    double gbytes = total_bytes / (1000.0 * 1000.0 * 1000.0); // NOTE: MB/sec and Gbps are based on 1000, GiB is based on 1024
    double Gbps = (gbytes * 8.0) / elapsed_seconds;
    accumulated_throughput_Gbps = (accumulated_throughput_Gbps < 0.0) ? Gbps : Common::smoothValue(accumulated_throughput_Gbps, Gbps, 0.1);
    double MBps = accumulated_throughput_Gbps / 8.0 * 1000.0; // NOTE: MB/sec and Gbps are based on 1000, GiB is based on 1024

    // See if time to print results
    double elapsed_print_seconds = end_time - print_start_time;
    if (elapsed_print_seconds >= Common::ELAPSED_SECONDS_TO_PRINT) {
      print_start_time = end_time;
      char message[300];
      snprintf(message, 300, "Provider: '%s',  Method: %s,  Message size: %lu bytes,  Iterations: %u,  Throughput: %7.3f Gbps (%0.1f MBps)",
               _app_params.provider.c_str(), _app_params.use_polling ? " polling" : "event-driven", _app_params.nbytes, _app_params.iters, Gbps, MBps);
      if (_app_params.run_forever) {
        printf("\r%s     ", message);
        fflush(stdout);
      } else {
        printf("%s\n", message);
      }
    }

  } while (_app_params.run_forever);

  // Finalize
  if (_app_params.verbose) printf("Finalizing throughput connection...\n");
  UK_RECORD_EVENT(_app_params.unikorn_session, FINALIZE_START_ID, 0);
  (void)takyonDestroy(path, TAKYON_WAIT_FOREVER);
  delete[] recv_sub_buffers;
  delete[] recv_requests;
  delete[] send_request_in_use;
  delete[] send_sub_buffers;
  delete[] send_requests;
  Common::freeTransportMemory(send_memory);
  Common::freeTransportMemory(recv_memory);
  UK_RECORD_EVENT(_app_params.unikorn_session, FINALIZE_END_ID, 0);

  UK_RECORD_EVENT(_app_params.unikorn_session, THROUGHPUT_TEST_END_ID, 0);
}
