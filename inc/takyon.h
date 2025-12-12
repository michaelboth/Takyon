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
#define TAKYON_API_VERSION_MAJOR 2
#define TAKYON_API_VERSION_MINOR 1

// Takyon text length limits
#define TAKYON_MAX_PROVIDER_CHARS    1000     // Max size of text string to define a path's provider
#define TAKYON_MAX_BUFFER_NAME_CHARS   31     // This small value of 31 is imposed by Apple's OSX mmap name limit

// Special timeout values
#define TAKYON_NO_WAIT       0  // Usually used to check an active transfer's status
#define TAKYON_WAIT_FOREVER -1  // Used to wait forever if needed for a transfer to complete

// Verbosities: a set of mask values that can or'ed together
#define TAKYON_VERBOSITY_NONE                0x00  // A convenience to show there is no verbosity, not even error messages
#define TAKYON_VERBOSITY_ERRORS              0x01  // If enabled, print all errors to strerr (otherwise error printing is suppressed)
#define TAKYON_VERBOSITY_CREATE_DESTROY      0x02  // Minimal stdout messages about path creation/destroying
#define TAKYON_VERBOSITY_CREATE_DESTROY_MORE 0x04  // Additional stdout messages about path creation/destroying
#define TAKYON_VERBOSITY_TRANSFERS           0x08  // Minimal stdout messages about active transfers
#define TAKYON_VERBOSITY_TRANSFERS_MORE      0x10  // Additional stdout messages about active transfers

// Error handling modes; i.e. what to do when Takyon detects and error
typedef enum {
  TAKYON_ABORT_ON_ERROR  = 0xBadFace,  // Print error and call abort()
  TAKYON_EXIT_ON_ERROR   = 0xBadCafe,  // Print error and call exit(EXIT_FAILURE)
  TAKYON_RETURN_ON_ERROR = 0xBadBabe   // Takyon will return from the function, and print the error if TAKYON_VERBOSITY_ERRORS is enabled
} TakyonFailureMode;

// One-sided transfer modes
typedef enum {
  TAKYON_OP_READ,                            // Read data from the remote endpoint (remote endpoint is not actively involed in transfer)
  TAKYON_OP_WRITE,                           // Write data to the remote endpoint (remote endpoint is not actively involed in transfer)
  TAKYON_OP_WRITE_WITH_PIGGYBACK,            // Write data to the remote endpoint. The remote endpoint must have a posted receive with
                                             // zero sub buffers in order to catch the 32bit piggyback message with takyonIsRecved().
                                             // The number of bytes written will be reported by takyonIsRecved().
  TAKYON_OP_ATOMIC_COMPARE_AND_SWAP_UINT64,  // This is a remote atomic operation:
                                             // if (remote_value == request.atomics[0]) {
                                             //   request.sub_buffers[0][0] = remote_value;
                                             //   remote_value = request.atomics[1];
                                             // }
  TAKYON_OP_ATOMIC_ADD_UINT64,               // This is a remote atomic operation:
                                             // {
                                             //   request.sub_buffers[0][0] = remote_value;
                                             //   remote_value += request.atomics[0];
                                             // }
} TakyonOneSidedOp;

// Defines a contiguous memory region, pre-allocated by the app, that Takyon transfer operations may use.
// In some cases, the memory defined in this buffer will be prepared for use; e.g. RDMA will pin it.
// IMPORTANT: App must maintain this memory for life of path.
typedef struct {
  void *addr;                              // Application must allocate the memory; CPU, CUDA, IO device, etc. The provider must support the type of memory allocated.
  uint64_t bytes;                          // Number of contiguous bytes in the buffer starting at 'addr'
  char name[TAKYON_MAX_BUFFER_NAME_CHARS]; // Only needed for special memory like inter-process mmaps. Ignore if not needed.
  void *app_data;                          // Application uses this as needed. Takyon does not look at this. Helpful for application dependent data; e.g. store an mmap handle or CUDA device ID.
  void *private_data;                      // Used internally by Takyon and should not be modified by application; e.g. registered memory info
} TakyonBuffer;

// These are used in transfer requests and represent a contiguous sub region of a TakyonBuffer.
// The memory represented by this should not be modifed while being sent/written, or read while being received/read.
typedef struct {
  uint32_t buffer_index;  // Index into the list of 'buffers' (TakyonBuffer) provided in TakyonPathAttributes
  uint64_t bytes;         // This can be less than the full Takyon buffer.
                          // IMPORTANT: Receiver can make this more than what is actually sent, takyonIsRecved() will report that actual bytes received.
                          //            This has the benefit of allowing the number of bytes sent to be dynamic, and the receiver does not need to know ahead of time.
  uint64_t offset;        // Offset into the Takyon buffer
  void *private_data;     // Used internally by Takyon and should not be modified by application; e.g. optimize posting receives
} TakyonSubBuffer;

// Only used for one-sided transfers: read/write/atomics
// IMPORTANT: App must maintain this structure for the life of the transfer, unless use_is_done_notification == false
typedef struct {
  TakyonOneSidedOp operation;
  // Local memory info
  uint32_t sub_buffer_count;                 // Some providers support > 1 (e.g. RDMA supports a mix of CUDA and CPU memory blocks)
  TakyonSubBuffer *sub_buffers;              // List of memory blocks in the local takyon buffer(s) defined with this path
  // Only for atomics
  uint64_t atomics[2];                       // See TAKYON_OP_ATOMIC_COMPARE_AND_SWAP_UINT64 and TAKYON_OP_ATOMIC_ADD_UINT64 above for details
  // Remote memory info
  uint32_t remote_buffer_index;              // Index into the remote endpoint's buffer list
  uint64_t remote_offset;                    // Offset in bytes into the remote endpoint's buffer addr
  // Completion fields
  bool submit_fence;                         // Forces preceding non-blocking transfers (send, read, write, atomics), where notification is turned off,
                                             // to complete before the new transfer starts. This is typically only needed if a preceding 'read' or 'atomic'
                                             // operation is invoked (changes local memory) just before sending the results of the preceding operations
  bool use_is_done_notification;             // If 'true' and takyonIsOneSidedDone() is supported, then must call takyonIsOneSidedDone()
  bool use_polling_completion;               // Use 'true' to use CPU polling to detect transfer completion. Great for low latency.
                                             // Use 'false' to use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts; // If use_polling_completion == true, then this defines the number of microseconds to sleep between poll attempts.
                                             // This helps to avoid burning up CPU when polling, but if full polling is needed, then set this to 0.
  // Helpful for application dependent data
  void *app_data;                            // Application uses this as needed. Takyon does not look at this.
  // Do not modify the following fields
  void *private_data;                        // Used internally by Takyon and should not be modified by application; e.g. track the completion between takyonOneSided() and takyonIsOneSidedDone()
} TakyonOneSidedRequest;

// Used only for sending in two-sided transfers (send ---> recv)
// IMPORTANT: App must maintain this structure for the life of the transfer, unless use_is_sent_notification == false
typedef struct {
  // The data
  uint32_t sub_buffer_count;                 // Some providers support > 1 (e.g. RDMA supports a mix of CUDA and CPU memory blocks)
                                             // Some providers support 0; good for zero byte messages, which are usually used for pseudo signaling.
  TakyonSubBuffer *sub_buffers;              // List to memory blocks in the local takyon buffer(s) defined with this path. Set to NULL if sub_buffer_count == 0.
  // Completion fields
  bool submit_fence;                         // Forces preceding non-blocking transfers (send, read, write, atomics), where notification is turned off,
                                             // to complete before the new transfer starts. This is typically only needed if a preceding 'read' or 'atomic'
                                             // operation is invoked (changes local memory) just before sending the results of the preceding operations
  bool use_is_sent_notification;             // If 'true' and takyonIsSent() is supported, then must call takyonIsSent()
  bool use_polling_completion;               // Use 'true' to use CPU polling to detect transfer completion. Great for low latency.
                                             // Use 'false' to use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts; // If use_polling_completion == true, then this defines the number of microseconds to sleep between poll attempts.
  // Helpful for application dependent data
  void *app_data;                            // Application uses this as needed. Takyon does not look at this
  // Do not modify the following fields
  void *private_data;                        // Used internally by Takyon and should not be modified by application; e.g. track the completion between takyonSend() and takyonIsSent()
} TakyonSendRequest;

// Used only for receiving in two-sided transfers (send ---> recv)
// IMPORTANT: App must maintain this structure for the life of the transfer
typedef struct {
  // Transfer info fields
  uint32_t sub_buffer_count;                   // Some providers support > 1 (e.g. RDMA supports a mix of CUDA and CPU memory blocks)
                                               // Some providers support 0; good for zero byte messages, which are usually used for pseudo signaling.
  TakyonSubBuffer *sub_buffers;                // List to memory blocks in the local takyon buffer(s) defined with this path. Set to NULL if sub_buffer_count == 0.
  // Completion fields
  bool use_polling_completion;                 // Use 'true' to use CPU polling to detect transfer completion. Great for low latency.
                                               // Use 'false' to use event driven (allows CPU to sleep) to passively detect completion.
  uint32_t usec_sleep_between_poll_attempts;   // If use_polling_completion == true, then this defines the number of microseconds to sleep between poll attempts.
  // Helpful for application dependent data
  void *app_data;                              // Application uses this as needed. Takyon does not look at this.
  // Do not modify the following fields
  void *private_data;                          // Used internally by Takyon and should not be modified by application; e.g. track the completion between takyonPostRecvs() and takyonIsRecved()
} TakyonRecvRequest;

// Define the behaviour of a path's endpoint.
// IMPORTANT: takyonCreate() will make a copy of this structure but the 'buffers' must be application allocated and persistant for the life of the app
typedef struct {
  bool is_endpointA;                           // If 'true' this is considiered side A of the path.
                                               // If 'false' this is considiered side B of the path.
  char provider[TAKYON_MAX_PROVIDER_CHARS];    // Text string the describes the endpoint's provider specification.
                                               // To know the supported provider sepcs, see the implementation's usage guide.
  uint64_t verbosity;                          // 'Or' the bits of the TAKYON_VERBOSITY_* mask values to define what is printed to stdout and stderr.
  TakyonFailureMode failure_mode;              // Determine what happens when the Takyon function detects an error
  // Transport buffers. Some providers pin this memory to avoid it being swapped out to RAM disk
  uint32_t buffer_count;                       // Number of contiguous memory buffers that this endpoint with use with transfers.
                                               // IMPORTANT: this can be 0 if no data will be transferred; i.e. only used for pseudo signaling
  TakyonBuffer *buffers;                       // List of buffers to prepare for message transfers
                                               // IMPORTANT: App must maintain this list of memory buffers for the life of the path
  // The following defines the max limits of how the local endpoint with be used (not how the remote endpoint, of the same path, will be used)
  uint32_t max_pending_send_requests;          // If takyonIsSent() is supported, this defines how many active 'send' transfers can be in progress
  uint32_t max_pending_recv_requests;          // If takyonPostRecvs() is supported, this defines how many active 'recv' transfers can be posted
  uint32_t max_pending_write_requests;         // If takyonIsOneSidedDone() and one_sided_write_supported is supported, this defines how many active 'write' transfers can be in progress
  uint32_t max_pending_read_requests;          // If takyonIsOneSidedDone() and one_sided_read_supported is supported, this defines how many active 'read' transfers can be in progress
  uint32_t max_pending_atomic_requests;        // If takyonIsOneSidedDone() and one_sided_atomics_supported is supported, this defines how many active 'atomic' transfers can be in progress
  uint32_t max_sub_buffers_per_send_request;   // Defines the number of sub buffers in a single 'send' message. Will be ignored if provider only supports 1.
  uint32_t max_sub_buffers_per_recv_request;   // Defines the number of sub buffers in a single 'recv' message. Will be ignored if provider only supports 1.
  uint32_t max_sub_buffers_per_write_request;  // Defines the number of sub buffers in a single 'write' message. Will be ignored if provider only supports 1.
  uint32_t max_sub_buffers_per_read_request;   // Defines the number of sub buffers in a single 'read' message. Will be ignored if provider only supports 1.
} TakyonPathAttributes;

// IMPORTANT: This structure is not user defined, but will be filled in by takyonCreate() and accessed in the returned path->capabilities.
//            It allows the application to intelligently define transfers based on the provider's capabilities.
typedef struct {
  // Two-sided functionality supported by provider
  bool Send_function_supported;
  bool IsSent_function_supported;         // Only supported when interconnect is non blocking (does not involve the CPU)
  bool PostRecvs_function_supported;      // Can be supported for blocking and non blocking interconnects, but not for stream interconnects (e.g. sockets)
  bool IsRecved_function_supported;
  // One-sided functionality supported by provider
  bool OneSided_function_supported;       // May only include a subset of read, write, atomic compare and swap, atomic add
  bool IsOneSidedDone_function_supported; // Only supported when interconnect is non blocking (does not involve the CPU)
  bool one_sided_write_supported;
  bool one_sided_read_supported;
  bool one_sided_atomics_supported;
  // Other characteristics supported by provider
  bool is_unreliable;                     // From the receiver's point of view, sent messages may be quietly dropped, arrive out of order, or be duplicated
  bool piggyback_messages_supported;      // True if comm allows sending a 32bit message piggybacked on the primary message
  bool multi_sub_buffers_supported;       // True if more than one sub buffer can be in a single transfer
  bool zero_byte_messages_supported;      // True if can send zero byte messages
} TakyonPathCapabilities;

// The primary path handle, filled in by takyonCreate(), and used by all other takyon functions
typedef struct {
  // IMPORTANT: Do not modify any fields in this structure
  TakyonPathAttributes attrs;           // Contains a copy of the attributes passed in from takyonCreate(), but does not copy contents of pointers
  TakyonPathCapabilities capabilities;  // This will be filled in by takyonCreate()
  char *error_message;                  // For returning error messages if a failure occurs with sending, or receiving. This should not be freed by the application.
  void *private_data;                   // Used internally by Takyon and should not be modified by application; e.g. endpoint book keeping
} TakyonPath;


#ifdef __cplusplus
extern "C"
{
#endif

// Create/Destroy a communication endpoint
// ---------------------------------------
//   If takyonCreate() returns non NULL, it contains the error message, and it needs to be freed by the application.
//   If takyonPostRecv() is not supported by the provider, then takyonCreate() will ignore post_recv_count and recv_requests.
extern char *takyonCreate(TakyonPathAttributes *attrs, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds, TakyonPath **path_out);
extern char *takyonDestroy(TakyonPath *path, double timeout_seconds);

// TWO-SIDED TRANSFERS
// -------------------
//   Both endpoints are involved in the transfer:
//     A:send --> B:recv
//     A:recv --> B:send
//   NOTE: If multicast is supported by the provider, then can have multiple recv endpoints for a single send endpoint.
//   NOTE: Two-sided transfers are designed to be one-way; i.e. no round trip needed to complete the transfer.
//   This group of functions returns true if there was an unrecoverable error.
extern bool takyonSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret);
extern bool takyonIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool takyonPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
extern bool takyonIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret);

// ONE-SIDED TRANSFERS
// -------------------
//   Only one endpoint is involved in the transfer; i.e. the other endpoint will not know the transfer is even occuring and not use any OS resources during the transfer.
//   Write message: one-way send (i.e. push data to the remote endpoint). If piggyback message is sent, remote endpoint must call takyonIsRecved() to get the piggy back message.
//     A:write --> (B not involved)
//     B:write --> (A not involved)
//   Read message: one way recv (i.e. pulling the data from the remote endpoint), no involvement from the remote endpoint
//     A:read <-- (B not involved)
//     B:read <-- (A not involved)
//   Atomic compare and swap uint64:
//     See TAKYON_OP_ATOMIC_COMPARE_AND_SWAP_UINT64 above for details
//   Atomic add uint64:
//     See TAKYON_OP_ATOMIC_ADD_UINT64 above for details
//   This group of functions returns true if there was an unrecoverable error.
extern bool takyonOneSided(TakyonPath *path, TakyonOneSidedRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret);
extern bool takyonIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);

#ifdef __cplusplus
}
#endif

#endif
