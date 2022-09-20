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
//   - Stripped out all the stuff that was provider specific
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

#ifndef _takyon_private_h_
#define _takyon_private_h_

#include "takyon.h"

// Some helpful values
#define MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING 20000  // 20 Milliseconds. Used by some providers to provided time to complete last transfer before disconnecting
#define NANOSECONDS_PER_SECOND_DOUBLE 1000000000.0        // Needed when converting timeouts from double seconds to int64_t nano_seconds
#define MAX_ERROR_MESSAGE_CHARS 1000

typedef struct {
  // Functions
  bool (*create)(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds);
  bool (*destroy)(TakyonPath *path, double timeout_seconds);
  bool (*oneSided)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*isOneSidedDone)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*send)(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret);
  bool (*isSent)(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*postRecvs)(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
  bool (*isRecved)(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

  // Use this maintain provider's instance data
  void *data;
} TakyonComm;

// Helpful global functionality
#define TAKYON_RECORD_ERROR(msg_ptr, ...) buildErrorMessage(msg_ptr, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#ifdef __cplusplus
extern "C"
{
#endif

// Error recording
extern void buildErrorMessage(char *error_message, const char *file, const char *function, int line_number, const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif
