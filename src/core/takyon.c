// Takyon 1.x was originally developed by Michael Both at Abaco, and he is now continuing development independently
//
// Original copyright:
//     Copyright 2018,2020 Abaco Systems
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.
//
// Changes for 2.0 (starting from Takyon 1.1.0):
//   - See comments in takyon.h for the bigger picture of the changes
//   - Redesigned to the 2.x API functionality
//
// Copyright for modifications:
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

#include "takyon_private.h"
#include "supported_providers.h"
#include "utils_arg_parser.h"
#include "utils_endian.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

#define MAX_PRIVATE_ERROR_MESSAGE_CHARS 10000               // This will hold the error stack message. Try to be terse!

static void handleErrorReporting(char *error_message, TakyonPathAttributes *attrs, const char *function) {
  // An error occured, so need to see how to handle it
  bool print_errors = (attrs->verbosity & TAKYON_VERBOSITY_ERRORS);
  if (attrs->failure_mode == TAKYON_ABORT_ON_ERROR || attrs->failure_mode == TAKYON_EXIT_ON_ERROR || print_errors) {
    // Make sure any normal prints are flushed
    fflush(stdout);
    // Start error message with endpoint identification
    if (print_errors) {
      fprintf(stderr, "ERROR in %s(), path=%s:'%s'\n", function, attrs->is_endpointA ? "A" : "B", attrs->provider);
    }
    // Print error message
    if (print_errors) {
      fprintf(stderr, "%s", error_message);
    }
    // Force the output
    fflush(stderr);
    // See if need to abort or exit
    if (attrs->failure_mode == TAKYON_ABORT_ON_ERROR) {
      abort();
    } else if (attrs->failure_mode == TAKYON_EXIT_ON_ERROR) {
      exit(EXIT_FAILURE);
    }
  }
  // Status will be returned
}

static void clearErrorMessage(char *error_message) {
  error_message[0] = '\0';
}

void buildErrorMessage(char *error_message, const char *file, const char *function, int line_number, const char *format, ...) {
  // IMPORTANT: make sure this only gets called in one thread and is not called by multiple threads (i.e. callback handlers)
  // NOTE: If comm is inter-thread, each endpoint allocates 'error'message' so no collision between them.
  va_list arg_ptr;
  // Create the message prefix
  size_t offset = strlen(error_message);
  snprintf(error_message+offset, MAX_PRIVATE_ERROR_MESSAGE_CHARS-offset, "  %s:%s():line=%d ", file, function, line_number);
  offset = strlen(error_message);
  // Add the error message
  va_start(arg_ptr, format);
  vsnprintf(error_message+offset, MAX_PRIVATE_ERROR_MESSAGE_CHARS-offset, format, arg_ptr);
  va_end(arg_ptr);
}

char *takyonCreate(TakyonPathAttributes *attrs, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds, TakyonPath **path_out) {
  // Validate arguments
  if (attrs == NULL) {
    fprintf(stderr, "ERROR in %s(): Argument 'attrs' is NULL\n", __FUNCTION__);
    abort();
  }
  if (attrs->failure_mode != TAKYON_ABORT_ON_ERROR && attrs->failure_mode != TAKYON_EXIT_ON_ERROR && attrs->failure_mode != TAKYON_RETURN_ON_ERROR) {
    fprintf(stderr, "ERROR in %s(): attrs->failure_mode must be one of TAKYON_ABORT_ON_ERROR, TAKYON_EXIT_ON_ERROR, or TAKYON_RETURN_ON_ERROR\n", __FUNCTION__);
    abort();
  }
  if (post_recv_count > 0 && recv_requests == NULL) {
    fprintf(stderr, "ERROR in %s(): Argument 'post_recv_count' is not 0, but 'recv_requests' is NULL\n", __FUNCTION__);
    abort();
  }
  if (post_recv_count == 0 && recv_requests != NULL) {
    fprintf(stderr, "ERROR in %s(): Argument 'post_recv_count' is 0, but 'recv_requests' is not NULL\n", __FUNCTION__);
    abort();
  }
  if (post_recv_count > attrs->max_pending_recv_requests) {
    fprintf(stderr, "ERROR in %s(): Argument 'post_recv_count' is greater than 'attrs.max_pending_recv_requests'\n", __FUNCTION__);
    abort();
  }
  if (path_out == NULL) {
    fprintf(stderr, "ERROR in %s(): Argument 'path_out' is NULL\n", __FUNCTION__);
    abort();
  }

  // Validate attributes
  if (strlen(attrs->provider) == 0 || strlen(attrs->provider) >= TAKYON_MAX_PROVIDER_CHARS) {
    fprintf(stderr, "ERROR in %s(): attrs->provider must not be empty and be less than TAKYON_MAX_PROVIDER_CHARS chars\n", __FUNCTION__);
    abort();
  }
  if (attrs->buffer_count == 0 && attrs->buffers != NULL) {
    fprintf(stderr, "ERROR in %s(): attrs->buffer_count is 0, so attrs->buffers must be NULL\n", __FUNCTION__);
    abort();
  }
  if (attrs->buffer_count > 0 && attrs->buffers == NULL) {
    fprintf(stderr, "ERROR in %s(): attrs->buffer_count is greater than 0, so attrs->buffers must not be NULL\n", __FUNCTION__);
    abort();
  }

  // Allocate error message string
  char *takyon_error_message = (char *)malloc(MAX_PRIVATE_ERROR_MESSAGE_CHARS);
  if (takyon_error_message == NULL) {
    fprintf(stderr, "ERROR in %s(): Failed to allocate error message memory\n", __FUNCTION__);
    abort();
  }
  clearErrorMessage(takyon_error_message);

  // Get the provider type
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(attrs->provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(takyon_error_message, "argGetProvider() failed: %s\n", error_message);
    handleErrorReporting(takyon_error_message, attrs, __FUNCTION__);
    return takyon_error_message;
  }

  // Allocate the path and is child structures
  TakyonPath *path = (TakyonPath *)calloc(1, sizeof(TakyonPath));
  if (path == NULL) {
    TAKYON_RECORD_ERROR(takyon_error_message, "Out of memory\n");
    handleErrorReporting(takyon_error_message, attrs, __FUNCTION__);
    return takyon_error_message;
  }

  // Create the private path
  TakyonComm *comm = (TakyonComm *)calloc(1, sizeof(TakyonComm));
  if (comm == NULL) {
    TAKYON_RECORD_ERROR(takyon_error_message, "Out of memory\n");
    free(path);
    handleErrorReporting(takyon_error_message, attrs, __FUNCTION__);
    return takyon_error_message;
  }

  // Fill in path capabilities and functions
  TakyonPathCapabilities capabilities;
  bool ok = setProviderFunctionsAndCapabilities(provider_name, comm, &capabilities);
  if (!ok) {
    TAKYON_RECORD_ERROR(takyon_error_message, "Provider '%s' is not found in 'supported_providers.c'\n", provider_name);
    free(comm);
    free(path);
    handleErrorReporting(takyon_error_message, attrs, __FUNCTION__);
    return takyon_error_message;
  }

  // Fill in the path
  path->attrs = *attrs;
  path->capabilities = capabilities;
  path->private = comm;
  path->error_message = takyon_error_message;

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY) {
    printf("%-15s (%s:%s) endian=%s, bits=%d\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, endianIsBig() ? "big" : "little", (int)sizeof(int*)*8);
  }

  // Initialize the actual connection (this is a blocking coordination between side A and B)
  ok = comm->create(path, post_recv_count, recv_requests, timeout_seconds);
  if (!ok) {
    TAKYON_RECORD_ERROR(takyon_error_message, "Failed to make connection with %s:%s\n", path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
    free(comm);
    free(path);
    handleErrorReporting(takyon_error_message, attrs, __FUNCTION__);
    return takyon_error_message;
  }

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY) {
    printf("%-15s (%s:%s) created path\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  *path_out = path;

  return NULL;
}

char *takyonDestroy(TakyonPath *path, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY) {
    printf("%-15s (%s:%s)\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  // If this is a connected path, do a coordinated shutdown with side A and B (this is a blocking call)
  bool ok = comm->destroy(path, timeout_seconds);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
  }

  // Check if there was an error
  char *error_message = path->error_message;
  if (ok) {
    free(error_message);
    error_message = NULL;
  }

  // Free resources
  free(comm);
  free(path);

  return error_message;
}

bool takyonOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);
  if (timed_out_ret != NULL) *timed_out_ret = false;

  // Error checking
  if (comm->oneSided == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonOneSided().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'request' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->local_buffer_index >= path->attrs.buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "'request->local_buffer_index' is out of range\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  TakyonBuffer *buffer = &path->attrs.buffers[request->local_buffer_index];
  if ((buffer->bytes - request->local_offset) < request->bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "'request->bytes' exceeds request->local_buffer range\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (timed_out_ret == NULL && timeout_seconds >= 0) {
    TAKYON_RECORD_ERROR(path->error_message, "If timeout_seconds >= 0 then 'timed_out_ret' must not be NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) %s message\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, request->is_write_request ? "Writing" : "Reading");
  }

  // Initiate the send
  bool ok = comm->oneSided(path, request, timeout_seconds, timed_out_ret);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (comm->isOneSidedDone == NULL && timed_out_ret != NULL && !(*timed_out_ret) && path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Message transferred\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  return true;
}

bool takyonIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);
  if (timed_out_ret != NULL) *timed_out_ret = false;

  // Error checking
  if (comm->isOneSidedDone == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonIsOneSidedDone().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'request' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->local_buffer_index >= path->attrs.buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "'request->local_buffer_index' is NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  TakyonBuffer *buffer = &path->attrs.buffers[request->local_buffer_index];
  if ((buffer->bytes - request->local_offset) < request->bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "'request->bytes' exceeds request->local_buffer range\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (timed_out_ret == NULL && timeout_seconds >= 0) {
    TAKYON_RECORD_ERROR(path->error_message, "If timeout_seconds >= 0 then 'timed_out_ret' must not be NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Waiting '%s message' to complete\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, request->is_write_request ? "write" : "read");
  }

  // Initiate the send
  bool ok = comm->isOneSidedDone(path, request, timeout_seconds, timed_out_ret);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (timed_out_ret != NULL && !(*timed_out_ret) && path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Message transferred\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  return true;
}

bool takyonSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);
  if (timed_out_ret != NULL) *timed_out_ret = false;

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Sending message\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  // Error checking
  if (comm->send == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonSend().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'request' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count > 1 && !path->capabilities.multi_sub_buffers_supported) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support request->sub_buffer_count > 1.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count >= 1 && request->sub_buffers == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count >= 1 but request->sub_buffers == NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count == 0 && request->sub_buffers != NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count == 0 but request->sub_buffers != NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (timed_out_ret == NULL && timeout_seconds >= 0) {
    TAKYON_RECORD_ERROR(path->error_message, "If timeout_seconds >= 0 then 'timed_out_ret' must not be NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Initiate the send
  /*+ bool timed_out; */
  bool ok = comm->send(path, request, piggy_back_message, timeout_seconds, timed_out_ret);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (comm->isSent == NULL && timed_out_ret != NULL && !(*timed_out_ret) && path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Message sent\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  return true;
}

bool takyonIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);
  if (timed_out_ret != NULL) *timed_out_ret = false;

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Waiting for message to be sent\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  // Error checking
  if (comm->isSent == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonIsSent().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'request' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (timed_out_ret == NULL && timeout_seconds >= 0) {
    TAKYON_RECORD_ERROR(path->error_message, "If timeout_seconds >= 0 then 'timed_out_ret' must not be NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Check for send completion
  bool ok = comm->isSent(path, request, timeout_seconds, timed_out_ret);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (timed_out_ret != NULL && !(*timed_out_ret) && path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Message sent\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  return true;
}

bool takyonPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Posting %d receive(s)\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, request_count);
  }

  // Error checking
  if (comm->postRecvs == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonPostRecvs().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request_count == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "'request_count' is 0.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (requests == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'requests' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *request = &requests[i];
    if (request->sub_buffer_count > 1 && !path->capabilities.multi_sub_buffers_supported) {
      TAKYON_RECORD_ERROR(path->error_message, "This provider does not support request->sub_buffer_count > 1.\n");
      handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
      return false;
    }
    if (request->sub_buffer_count >= 1 && request->sub_buffers == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count >= 1 but request->sub_buffers == NULL\n");
      handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
      return false;
    }
    if (request->sub_buffer_count == 0 && request->sub_buffers != NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count == 0 but request->sub_buffers != NULL\n");
      handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
      return false;
    }
  }

  // Prepare the recv
  bool ok = comm->postRecvs(path, request_count, requests);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  return true;
}

bool takyonIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  clearErrorMessage(path->error_message);
  if (timed_out_ret != NULL) *timed_out_ret = false;

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    printf("%-15s (%s:%s) Waiting for message\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider);
  }

  // Error checking
  if (comm->isRecved == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support takyonIsRecved().\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "'request' is NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (timed_out_ret == NULL && timeout_seconds >= 0) {
    TAKYON_RECORD_ERROR(path->error_message, "If timeout_seconds >= 0 then 'timed_out_ret' must not be NULL.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count > 1 && !path->capabilities.multi_sub_buffers_supported) {
    TAKYON_RECORD_ERROR(path->error_message, "This provider does not support request->sub_buffer_count > 1.\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count >= 1 && request->sub_buffers == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count >= 1 but request->sub_buffers == NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  if (request->sub_buffer_count == 0 && request->sub_buffers != NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "request->sub_buffer_count == 0 but request->sub_buffers != NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  /*+ use this
  if (piggy_back_message_ret != NULL && !path->capabilities.piggy_back_messages_supported) {
    TAKYON_RECORD_ERROR(path->error_message, "Piggy back messages are not supported with this provider, so piggy_back_message_ret must be NULL\n");
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }
  */

  // Wait for the message to arrive
  uint64_t bytes_received;
  uint32_t piggy_back_message;
  bool ok = comm->isRecved(path, request, timeout_seconds, timed_out_ret, &bytes_received, &piggy_back_message);
  if (!ok) {
    handleErrorReporting(path->error_message, &path->attrs, __FUNCTION__);
    return false;
  }

  // Verbosity
  if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS) {
    if (path->capabilities.piggy_back_messages_supported) {
      printf("%-15s (%s:%s) Got message: " UINT64_FORMAT " bytes, piggy_back_message=0x%x\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, bytes_received, piggy_back_message);
    } else {
      printf("%-15s (%s:%s) Got message: " UINT64_FORMAT " bytes\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, bytes_received);
    }
  }

  // Return info
  if (bytes_received_ret != NULL) *bytes_received_ret = bytes_received;
  if (piggy_back_message_ret != NULL) *piggy_back_message_ret = piggy_back_message;

  return true;
}
