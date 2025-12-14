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

#include "LatencyTest.hpp"
#ifdef ENABLE_CUDA
  #include "LatencyTestKernels.hpp"
#endif
#include "takyon.h"
#include "unikorn_instrumentation.h"
#include <cstring>

// Algorithm:
//   - Use a single buffer sender and receiver
//
// Algorithm:
//   loop {
//     if (is_sender) {
//       start send
//       recv & repost
//       waitForSendCompletion
//     } else {  // is_receiver
//       recv & repost
//       send
//       waitForSendCompletion
//     }
//   }
//
//  For most accurate results use 4 bytes messages.
//  For best results use polling with no sleep delay, since event-driven completion requires a thread based context switch

static void waitForMessage(TakyonPath *_path, TakyonRecvRequest *_recv_request, uint64_t _expected_bytes, uint32_t _expected_piggyback_message) {
  // Wait for message
  uint64_t bytes_received = 0;
  uint32_t piggyback_message = 0;
  (void)takyonIsRecved(_path, _recv_request, TAKYON_WAIT_FOREVER, NULL, &bytes_received, &piggyback_message);
  if (piggyback_message != _expected_piggyback_message) { EXIT_WITH_MESSAGE(std::string("Received message piggyback should be " + std::to_string(_expected_piggyback_message) + " but got " + std::to_string(piggyback_message))); }
  if (bytes_received != _expected_bytes) { EXIT_WITH_MESSAGE(std::string("Received message bytes should be " + std::to_string(_expected_bytes) + " but got " + std::to_string(bytes_received))); }

  // Re-post the recv
  if (_path->capabilities.PostRecvs_function_supported) {
    uint32_t request_count = 1;
    (void)takyonPostRecvs(_path, request_count, _recv_request);
  }
}

static void fillInValidationData(uint32_t *_buffer, uint64_t _count, uint32_t _starting_value, Common::MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) { _buffer[i] = (uint32_t)(_starting_value+i); }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU) {
    LatencyTestKernels::runFillInValidationDataKernelBlocking(_buffer, _count, _starting_value);
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
    LatencyTestKernels::runFillInValidationDataKernelBlocking(_buffer, _count, _starting_value);
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    /*+*/EXIT_WITH_MESSAGE(std::string("Not yet implemented"));
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}

static void copyValidationData(uint32_t *_input_buffer, uint32_t *_output_buffer, uint64_t _count, Common::MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) { _output_buffer[i] = _input_buffer[i]; }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU) {
    LatencyTestKernels::runCopyKernelBlocking(_input_buffer, _output_buffer, _count);
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
    LatencyTestKernels::runCopyKernelBlocking(_input_buffer, _output_buffer, _count);
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    /*+*/EXIT_WITH_MESSAGE(std::string("Not yet implemented"));
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}

static void validateData(uint32_t *_buffer, uint64_t _count, uint32_t _starting_value, Common::MemoryType _memory_type) {
  if (_memory_type == Common::MemoryType::CPU) {
    for (uint64_t i=0; i<_count; i++) {
      uint32_t expected_value = (uint32_t)(_starting_value+i);
      if (_buffer[i] != expected_value) {
        EXIT_WITH_MESSAGE(std::string("Received invalid data, _starting_value=" + std::to_string(_starting_value) + ", at index " + std::to_string(i) + " expected  " + std::to_string(expected_value) + " but got " + std::to_string(_buffer[i])));
      }
    }
#ifdef ENABLE_CUDA
  } else if (_memory_type == Common::MemoryType::SocIntegratedGPU) {
    /*+ reduction kernel */
    for (uint64_t i=0; i<_count; i++) {
      uint32_t expected_value = (uint32_t)(_starting_value+i);
      if (_buffer[i] != expected_value) {
        EXIT_WITH_MESSAGE(std::string("Received invalid data, _starting_value=" + std::to_string(_starting_value) + ", at index " + std::to_string(i) + " expected  " + std::to_string(expected_value) + " but got " + std::to_string(_buffer[i])));
      }
    }
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withGPUDirect) {
    /*+*/EXIT_WITH_MESSAGE(std::string("Not yet implemented"));
  } else if (_memory_type == Common::MemoryType::DiscreteGPU_withoutGPUDirect) {
    /*+*/EXIT_WITH_MESSAGE(std::string("Not yet implemented"));
#endif
  } else {
    EXIT_WITH_MESSAGE(std::string("To use GPU memory, the app needs to be built with CUDA"));
  }
}

void LatencyTest::runLatencyTest(bool _is_sender, const Common::AppParams &_app_params) {
  UK_RECORD_EVENT(_app_params.unikorn_session, LATENCY_TEST_START_ID, 0);

  // Determine the type of memory to use for processing and transport
  bool is_for_rdma = (_app_params.provider == "RC" || _app_params.provider == "UC" || _app_params.provider == "UD");
  Common::MemoryType memory_type = Common::memoryTypeToUseForTransport(is_for_rdma, _app_params.is_for_gpu);

  // Allocate transport memory
  uint64_t message_bytes = _app_params.nbytes;
  uint64_t total_send_bytes = message_bytes;
  uint64_t extra_recv_bytes = (_app_params.provider == "UD") ? 40 : 0; // RDMA UD receiver needs to allocate 40 extra bytes for RDMA's global routing header
  uint64_t total_recv_bytes = message_bytes + extra_recv_bytes;
  void *send_memory = Common::allocateTransportMemory(total_send_bytes, memory_type);
  void *recv_memory = Common::allocateTransportMemory(total_recv_bytes, memory_type);

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

  // Setup the recv request
  TakyonSubBuffer recv_sub_buffer;
  TakyonRecvRequest recv_request;
  {
    // Sub buffer
    recv_sub_buffer.buffer_index = 1; // Second transport buffer
    recv_sub_buffer.bytes = message_bytes + extra_recv_bytes;
    recv_sub_buffer.offset = 0;
    // Request
    recv_request.sub_buffer_count = 1;
    recv_request.sub_buffers = &recv_sub_buffer;
    recv_request.use_polling_completion = _app_params.use_polling;
    recv_request.usec_sleep_between_poll_attempts = 0;  // Use 0 since this test requires full speed completion monitoring to minimize latency
    recv_request.app_data = NULL;
  }

  // Setup the send requests
  TakyonSubBuffer send_sub_buffer;
  TakyonSendRequest send_request;
  {
    // Sub buffer
    send_sub_buffer.buffer_index = 0; // First transport buffer
    send_sub_buffer.bytes = message_bytes;
    send_sub_buffer.offset = 0;
    // Request
    send_request.sub_buffer_count = 1;
    send_request.sub_buffers = &send_sub_buffer;
    send_request.submit_fence = false; // Not needed since not using read or atomics
    send_request.use_is_sent_notification = true; // This needs to be true, since the recv notification might arrive before send completion notification is received
    send_request.use_polling_completion = _app_params.use_polling;
    send_request.usec_sleep_between_poll_attempts = 0;  // Use 0 since this test requires full speed completion monitoring to minimize latency
    send_request.app_data = NULL;
  }

  // Setup Takyon attributes
  TakyonPathAttributes attrs;
  strncpy(attrs.provider, _app_params.provider_params.c_str(), TAKYON_MAX_PROVIDER_CHARS-1);
  attrs.is_endpointA                      = _is_sender;
  attrs.failure_mode                      = TAKYON_EXIT_ON_ERROR;
  attrs.verbosity                         = TAKYON_VERBOSITY_ERRORS; //  | TAKYON_VERBOSITY_CREATE_DESTROY | TAKYON_VERBOSITY_CREATE_DESTROY_MORE | TAKYON_VERBOSITY_TRANSFERS | TAKYON_VERBOSITY_TRANSFERS_MORE;
  attrs.buffer_count                      = 2;
  attrs.buffers                           = transport_buffers;
  attrs.max_pending_send_requests         = 1;
  attrs.max_pending_recv_requests         = 1;
  attrs.max_pending_write_requests        = 0; // Not used
  attrs.max_pending_read_requests         = 0; // Not used
  attrs.max_pending_atomic_requests       = 0; // Not used
  attrs.max_sub_buffers_per_send_request  = 1;
  attrs.max_sub_buffers_per_recv_request  = 1;
  attrs.max_sub_buffers_per_write_request = 0; // Not used
  attrs.max_sub_buffers_per_read_request  = 0; // Not used

  // Init
  if (_app_params.verbose) printf("Initializing latency test connection...\n");
  uint32_t post_recv_count = 1;
  TakyonPath *path = NULL;
  (void)takyonCreate(&attrs, post_recv_count, &recv_request, TAKYON_WAIT_FOREVER, &path);

  // Throught main loop
  if (_app_params.verbose) printf("Collecting latency results...\n");
  double *round_trip_results_usecs = new double[_app_params.iters];
  double start_time = Common::clockTimeSeconds();
  double print_start_time = -Common::ELAPSED_SECONDS_TO_PRINT; // Make sure first pass prints
  double accumulated_latency_usecs = -1.0;
  double accumulated_jitter_usecs = -1.0;
  uint32_t *int_send_buffer = (uint32_t *)send_memory;
  uint32_t *int_recv_buffer = (uint32_t *)((uint64_t)recv_memory + extra_recv_bytes);
  uint32_t counter = 0;
  uint64_t validation_count = message_bytes/sizeof(uint32_t);
  uint32_t send_piggyback_message = 0; // Used for validation
  uint32_t recv_piggyback_message = 1000000000; // Used for validation
  do {
    for (uint32_t iter=0; iter<_app_params.iters; iter++) {
      UK_RECORD_EVENT(_app_params.unikorn_session, LATENCY_ITERATION_START_ID, 0);

      if (_is_sender) {
        // Send
        if (_app_params.validate) {
          fillInValidationData(int_send_buffer, validation_count, counter, memory_type);
        }
        send_piggyback_message++;
        (void)takyonSend(path, &send_request, send_piggyback_message, TAKYON_WAIT_FOREVER, NULL);
        // Recv
        recv_piggyback_message++;
        waitForMessage(path, &recv_request, message_bytes, recv_piggyback_message);
        if (_app_params.validate) {
          validateData(int_recv_buffer, validation_count, counter, memory_type);
        }
        // Make sure send completion has occurred
        if (path->capabilities.IsSent_function_supported) {
          (void)takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
        }

      } else {
        // Recv
        send_piggyback_message++;
        waitForMessage(path, &recv_request, message_bytes, send_piggyback_message);
        if (_app_params.validate) {
          validateData(int_recv_buffer, validation_count, counter, memory_type);
        }
        // Send
        if (_app_params.validate) {
          copyValidationData(int_recv_buffer, int_send_buffer, validation_count, memory_type);
        }
        recv_piggyback_message++;
        (void)takyonSend(path, &send_request, recv_piggyback_message, TAKYON_WAIT_FOREVER, NULL);
        // Make sure send completion has occurred
        if (path->capabilities.IsSent_function_supported) {
          (void)takyonIsSent(path, &send_request, TAKYON_WAIT_FOREVER, NULL);
        }
      }

      UK_RECORD_EVENT(_app_params.unikorn_session, LATENCY_ITERATION_END_ID, 0);

      double end_time = Common::clockTimeSeconds();
      round_trip_results_usecs[iter] = (end_time - start_time) * 1000000.0; // Conter to usecs
      start_time = end_time;

      counter++;
    }

    // Calculate best latency and worse jitter (start half way through the list to avoid warm up delays)
    double best_latency_usecs = round_trip_results_usecs[0];
    double worse_latency_usecs = best_latency_usecs;
    for (uint32_t iter=0; iter<_app_params.iters; iter++) {
      if (round_trip_results_usecs[iter] < best_latency_usecs) best_latency_usecs = round_trip_results_usecs[iter];
      if (round_trip_results_usecs[iter] > worse_latency_usecs) worse_latency_usecs = round_trip_results_usecs[iter];
    }
    double latency_usecs = best_latency_usecs/2.0;
    double jitter_usecs = (worse_latency_usecs - best_latency_usecs) / 2.0;
    accumulated_latency_usecs = (accumulated_latency_usecs < 0.0) ? latency_usecs : Common::smoothValue(accumulated_latency_usecs, latency_usecs, 0.1);
    accumulated_jitter_usecs = (accumulated_jitter_usecs < 0.0) ? jitter_usecs : Common::smoothValue(accumulated_jitter_usecs, jitter_usecs, 0.1);

    // See if time to print results
    double end_time = Common::clockTimeSeconds();
    double elapsed_seconds = end_time - print_start_time;
    if (elapsed_seconds >= Common::ELAPSED_SECONDS_TO_PRINT) {
      print_start_time = end_time;
      char message[300];
      snprintf(message, 300, "Provider: '%s',  Method: %s,  Message size: " UINT64_FORMAT " bytes,  Memory: %s,  Iterations: %u,  Latency: %6.2f usecs,  Jitter: %6.2f usecs",
               _app_params.provider.c_str(), _app_params.use_polling ? " polling" : "event-driven", message_bytes, memoryTypeToText(memory_type).c_str(), _app_params.iters, accumulated_latency_usecs, accumulated_jitter_usecs);
      if (_app_params.run_forever) {
        printf("\r%s     ", message);
        fflush(stdout);
      } else {
        printf("%s\n", message);
      }
    }

  } while (_app_params.run_forever);

  // Finalize
  if (_app_params.verbose) printf("Finalizing latency test connection...\n");
  (void)takyonDestroy(path, TAKYON_WAIT_FOREVER);
  delete[] round_trip_results_usecs;
  Common::freeTransportMemory(send_memory, memory_type);
  Common::freeTransportMemory(recv_memory, memory_type);
  if (_app_params.verbose) printf("Latency test Done.\n\n");

  UK_RECORD_EVENT(_app_params.unikorn_session, LATENCY_TEST_END_ID, 0);
}
