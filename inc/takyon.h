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
//   - There will be no backward compatibility to version 1.x
//   - Ground-up redesign of the API to expose the power and flexibility of RDMA
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

#ifndef _takyon_h_
#define _takyon_h_

#include <stdbool.h>
#include <stdint.h>

// Takyon version
#define TAKYON_VERSION_MAJOR 2
#define TAKYON_VERSION_MINOR 0
#define TAKYON_VERSION_PATCH 0

// For comm's interconnect text
#define TAKYON_MAX_INTERCONNECT_CHARS 1000  // Max size of text string to define a path's interconnect
#define TAKYON_MAX_BUFFER_NAME_CHARS 31     // This small value of 31 is imposed by Apple's OSX mmap name limit

// Special timeout values
#define TAKYON_NO_WAIT       0  // Use as a timeout of zero seconds
#define TAKYON_WAIT_FOREVER -1  // Use as a flag to never timeout waiting from a transfer to complete

// Verbosities: a set of mask values that can or'ed together
#define TAKYON_VERBOSITY_NONE                0x00  // A convenience to show there is no verbosity, not even error messages
#define TAKYON_VERBOSITY_ERRORS              0x01  // If enabled, print all errors to strerr (otherwise error printing is suppressed)
#define TAKYON_VERBOSITY_CREATE_DESTROY      0x02  // Minimal stdout messages about path creation/destroying
#define TAKYON_VERBOSITY_CREATE_DESTROY_MORE 0x04  // Additional stdout messages about path creation/destroying
#define TAKYON_VERBOSITY_TRANSFER            0x08  // Minimal stdout messages about active transfers
#define TAKYON_VERBOSITY_TRANSFER_MORE       0x10  // Additional stdout messages about active transfers

// Error handling modes
typedef enum {
  TAKYON_ABORT_ON_ERROR  = 0xBadFace,  // Print error and call abort()
  TAKYON_EXIT_ON_ERROR   = 0xBadCafe,  // Print error and call exit(EXIT_FAILURE)
  TAKYON_RETURN_ON_ERROR = 0xBadBabe  // Takyon will return from the function, and print the error if TAKYON_VERBOSITY_ERRORS is enabled
} TakyonFailureMode;

// IMPORTANT: The following data structures are only valid for they were created in

// App must maintain this memory for life of path
typedef struct {
  void *addr;                              // Application must allocate the memory; CPU, CUDA, IO device, etc. The interconnect must support the type of memory allocated.
  uint64_t bytes;
  char name[TAKYON_MAX_BUFFER_NAME_CHARS]; // Only needed for special memory like inter-process mmaps. Ignore if not needed
  // Helpful for application dependent data; e.g. store an mmap or CUDA device ID
  void *app_data;                          // Application uses this as needed. Takyon does not look at this
  // Do not modify the following fields
  void *private;                           // Used internally; e.g. registered memory info
} TakyonBuffer;

// Only used with multi buffer transfers
typedef struct {
  TakyonBuffer *buffer;
  uint64_t bytes;       // Receiver can make this more than what is actually sent, takyonIsRecved() will report that actual bytes received
  uint64_t offset;      // In bytes
  // Do not modify the following fields
  void *private;        // Used internally; e.g. optimize posting receives
} TakyonSubBuffer;

// App must maintain this structure for the life of the transfer
typedef struct {
  bool is_write_request;                     // True: is one sided write. False: is one sided read. Either way, the remote CPU is not involved in the transfer
  // Transfer info fields
  TakyonBuffer *local_buffer;                // The local memory buffer
  uint64_t local_offset;                     // Offset in bytes into the local buffer addr
  uint32_t remote_buffer_index;              // Index into the remote buffer list
  uint64_t remote_offset;                    // Offset in bytes into the buffer addr
  uint64_t bytes;                            // Byte to send from the buffer
  // Non blocking options
  bool use_is_done_notification;             // If true and takyonIsOneSidedDone() is supported, then must call takyonIsOneSidedDone()
  // Completion fields
  bool use_polling_completion;               // True: use CPU polling to detect transfer completion. False: use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts; // Use to avoid burning up CPU when polling
  // Helpful for application dependent data
  void *app_data;                            // Application uses this as needed. Takyon does not look at this
  // Do not modify the following fields
  void *private;                             // Used internally; e.g. track the completion between takyonOneSided() and takyonIsOneSidedDone()
} TakyonOneSidedRequest;

// App must maintain this structure for the life of the transfer
typedef struct {
  // Transfer info fields
  uint32_t sub_buffer_count;                 // Most comms will only support 0 or 1, but some will support > 1 (e.g. a mix of CUDA and CPU memory blocks)
  TakyonSubBuffer *sub_buffers;
  // Non blocking options
  bool use_is_sent_notification;             // If true and takyonIsSent() is supported, then must call takyonIsSent()
  // Completion fields
  bool use_polling_completion;               // True: use CPU polling to detect transfer completion. False: use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts; // Use to avoid burning up CPU when polling
  // Helpful for application dependent data
  void *app_data;                            // Application uses this as needed. Takyon does not look at this
  // Do not modify the following fields
  void *private;                             // Used internally; e.g. track the completion between takyonSend() and takyonIsSent()
} TakyonSendRequest;

// App must maintain this structure for the life of the transfer
typedef struct {
  // Transfer info fields
  uint32_t sub_buffer_count;                 // Most comms will only support 0 or 1, but some will support > 1 (e.g. a mix of CUDA and CPU memory blocks)
  TakyonSubBuffer *sub_buffers;
  // Completion fields
  bool use_polling_completion;               // True: use CPU polling to detect transfer completion. False: use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts; // Use to avoid burning up CPU when polling
  // Helpful for application dependent data
  void *app_data;                            // Application uses this as needed. Takyon does not look at this
  // Do not modify the following fields
  void *private;                             // Used internally; e.g. track the completion between takyonSend() and takyonIsSent()
} TakyonRecvRequest;

typedef struct {
  bool is_endpointA;                                 // True: side A of the path. False: side B of the path.
  char interconnect[TAKYON_MAX_INTERCONNECT_CHARS];  // Text string the describes the endpoint's interconnect specification.
  uint64_t verbosity;                                // 'Or' the bits of the TAKYON_VERBOSITY_* mask values to define what is printed to stdout and stderr.
  TakyonFailureMode failure_mode;                    // Determine what happens when an error is detected
  // Transport buffers. Some interconnects will pin this memory to avoid it being swapped out to RAM disk
  uint32_t buffer_count;
  TakyonBuffer *buffers;                             // App must maintain this memory for the life of the path
  // Used for internal book keeping and to avoid internal memory allocations at transfer time
  uint32_t max_pending_send_and_one_sided_requests;  // If takyonIsSent() and/or takyonIsOneSidedDone() is supported, this defines how many active transfers can be in progress
  uint32_t max_pending_recv_requests;                // If takyonPostRecvs() is supported, this defines how many active recvs can be posted at once
  uint32_t max_sub_buffers_per_send_request;         // Defines the number of sub buffers in a single send message. Will be ignored if interconnect only support 1.
  uint32_t max_sub_buffers_per_recv_request;         // Defines the number of sub buffers in a single recv message. Will be ignored if interconnect only support 1.
} TakyonPathAttributes;

typedef struct {
  // Functions
  bool OneSided_supported;
  bool IsOneSidedDone_supported;
  bool Send_supported;
  bool IsSent_supported;
  bool PostRecvs_supported;
  bool IsRecved_supported;
  // Extra features
  bool piggy_back_message_supported;      // True if comm allows sending a 32bit message piggy backed on the primary message
  bool multi_sub_buffers_supported;       // True if more than one sub buffer can be in a single transfer
  bool zero_byte_message_supported;       // True if can send zero byte messages
} TakyonPathFeatures;

typedef struct {
  // IMPORTANT: do not modify any fields in this structure
  TakyonPathAttributes attrs;   // Contains a copy of the attributes passed in from takyonCreate(), but does not copy contents of pointers
  TakyonPathFeatures features;
  char *error_message;                // For returning error messages if a failure occurs with sending, or receiving. This should not be freed by the application.
  void *private;                      // Used internally; e.g. book keeping
} TakyonPath;


#ifdef __cplusplus
extern "C"
{
#endif

// Create/destroy a communication endpoint
// If this returns non NULL, it contains the error message, and it needs to be freed by the application.
// If takyonPostRecv() is not supported, then post_recv_count and recv_requests will be ignored
extern char *takyonCreate(TakyonPathAttributes *attrs, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds, TakyonPath **path_out);
extern char *takyonDestroy(TakyonPath *path, double timeout_seconds);

// Write message: one way send (i.e. push data to the remote endpoint), no involvement from the remote endpoint
//   A --> (B not involved)
//   B --> (A not involved)
// Read message: one way recv (i.e. pulling the data from the remote endpoint), no involvement from the remote endpoint
//   A <-- (B not involved)
//   B <-- (A not involved)
extern bool takyonOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool takyonIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);

// Send a message (both endpoints are involved in the transfer)
// A --> B
// A <-- B
// Messages sent are received on the order the receives were posted
// 'piggy_back_message' must be 0 if 'features->piggy_back_message_supported' is false
extern bool takyonSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret);
extern bool takyonIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool takyonPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
extern bool takyonIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

#ifdef __cplusplus
}
#endif

#endif