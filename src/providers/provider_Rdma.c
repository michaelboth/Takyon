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

#include "provider_Rdma.h"
#include "takyon_private.h"
#include "utils_rdma_verbs.h"
#include "utils_socket.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include "utils_endian.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Supported formats:
//   RDMA RC (Reliable connected)
//     - Max message size is 1 GB
//     - Messages are reliable
//   RDMA UC (Unreliable connected)
//     - Max message size is 1 GB
//     - Messages can be dropped
//     - Useful where all bytes may not arrive (unreliable): e.g. live stream video or music
//   RDMA UD (Unreliable datagram)
//     - Max message size is RDMA MTU (4KB)
//     - Messages can be dropped
//     - Useful where all bytes may not arrive (unreliable): e.g. live stream video or music
//   ---------------------------------------------------------------------------
//     "RdmaRC -client -remoteIP=<ip_addr>|<hostname> -port=<port_number> -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//     "RdmaRC -server -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse] -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//
//     "RdmaUC -client -remoteIP=<ip_addr>|<hostname> -port=<port_number> -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//     "RdmaUC -server -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse] -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//
//     "RdmaUDUnicastSend -client -remoteIP=<ip_addr>|<hostname> -port=<port_number> -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//     "RdmaUDUnicastRecv -server -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse] -rdmaDevice=<name> -rdmaPort=<local_rdma_port_number> [<optional_args_see_below>]"
//
//   Argument descriptions:
//     -localIP=<ip_addr>|<hostname>|Any     Does not need to be the RDMA network interface
//     -remoteIP=<ip_addr>|<hostname>        Does not need to be the RDMA network interface; just needs to be at the same endpoint as the RDMA interface
//     -port=<port_number>                   [1024 .. 65535]
//     -rdmaDevice=<name>                    Name of the RDMA port; get from running the CLI: ibv_devinfo
//     -rdmaPort=<local_rdma_port_number>    1 .. max ports on local RDMA NIC; get from running the CLI: ibv_devinfo
//
//   Optional arguments:
//     -mtuBytes=<n>:                     RDMA MTU is less than network MTU. Default is detect min at endpoints (does not detect min in intermediate switches. Valid values are 256, 512, 1024, 2048, and 4096.
//     -gidIndex=<index>:                 RDMA's global address index. Default is 0.
//     -serviceLevel=<n>:                 Network quality of service level. Default is 0.
//     -hopLimit=<n>:                     Max routers to travel through. Default is 1.
//     -retransmitTimeout=<n>:   RC Only. Time to verify packet transmission (ACK or NACK). Default is 14. Range is [0 .. 31], for value meanings, see www.rdmamojo.com/2013/01/12/ibv_modify_qp/
//     -retryCnt=<n>:            RC Only. Retransmit attempts (without a NACK) before erroring. Default is 7, and max is 7.
//     -rnrRetry=<n>:            RC Only. Retransmit attempts (due to NACKs) before erroring. Default is 6. 7 is the max, but means infinite.
//     -minRnrTimer=<n>:         RC only. Incoming receive not ready. Default is 12. Range is [0 .. 31], for value meanings, see www.rdmamojo.com/2013/01/12/ibv_modify_qp/
//
//   Notes:
//     - RoCE v2 implicitly uses IP port 4791 for UD destination communications (unicast is unconnected)

#define MY_MAX(_a, _b) ((_a)>(_b) ? (_a) : (_b))

typedef struct {
  uint64_t bytes;
  uint64_t remote_addr;
  uint32_t remote_key;
} RemoteTakyonBuffer;

typedef struct {
  // Socket
  int socket_fd;
  bool thread_started;
  pthread_t disconnect_detection_thread_id;
  bool connection_failed;
  int read_pipe_fd;
  int write_pipe_fd;

  // Rdma
  RdmaEndpoint *endpoint;
  RdmaBuffer *rdma_buffer_list;
  uint32_t curr_rdma_send_request_index;
  uint32_t curr_rdma_recv_request_index;
  RdmaSendRequest *rdma_send_request_list;
  RdmaRecvRequest *rdma_recv_request_list;
  struct ibv_sge *send_sge_list;
  struct ibv_sge *recv_sge_list;

  // Remote buffers
  uint32_t remote_buffer_count;
  RemoteTakyonBuffer *remote_buffers;
} PrivateTakyonPath;

static void *disconnectDetectionThread(void *user_data) {
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)user_data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  // Wait for either a socket disconnect, or for takyonDestroy() to get called
  uint32_t dummy;
  int64_t timeout_nano_seconds = -1; // Wait forever
  bool timed_out = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  if (!socketRecv(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
    endpoint->connection_broken = true;
    pipeWakeUpPollFunction(private_path->write_pipe_fd, error_message, MAX_ERROR_MESSAGE_CHARS); // Wake up poll() in RDMA's completion event handler: eventDrivenCompletionWait()
  }
  // IMPORTANT: both endpoint need to be coordinated for this to work. If one endpoint does extra communications (UC recv), then this shut down process may block
  /*+ see if can find a way to gracefully hand RDMA UC recvs never getting data due to dropped messages */
  return NULL;
}

static bool sendRdmaBufferInfo(TakyonPath *path, PrivateTakyonPath *private_path, int64_t timeout_nano_seconds) {
  bool is_big_endian = endianIsBig();
  uint32_t buffer_count = path->attrs.buffer_count;
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  bool timed_out = false;

  // Send endian
  if (!socketSend(private_path->socket_fd, &is_big_endian, sizeof(is_big_endian), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to send is_big_endian: %s\n", error_message);
    return false;
  }
  // Send buffer count
  if (!socketSend(private_path->socket_fd, &buffer_count, sizeof(buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to send buffer_count: %s\n", error_message);
    return false;
  }

  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private_data;
    uint64_t bytes = buffer->bytes;
    uint64_t remote_addr = (uint64_t)buffer->addr; // This is the local address on this endpoint, but will be the remote addr on the remote endpoint
    uint32_t remote_key = rdma_buffer->mr->rkey;

    // Send bytes
    if (!socketSend(private_path->socket_fd, &bytes, sizeof(bytes), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send bytes: %s\n", error_message);
      return false;
    }
    // Send remote_addr
    if (!socketSend(private_path->socket_fd, &remote_addr, sizeof(remote_addr), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send remote_addr: %s\n", error_message);
      return false;
    }
    // Send remote_key
    if (!socketSend(private_path->socket_fd, &remote_key, sizeof(remote_key), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send remote_key: %s\n", error_message);
      return false;
    }
  }

  return true;
}

static bool recvRdmaBufferInfo(TakyonPath *path, PrivateTakyonPath *private_path, int64_t timeout_nano_seconds) {
  bool is_big_endian = endianIsBig();
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  bool timed_out = false;

  // Recv endian
  bool remote_is_big_endian;
  if (!socketRecv(private_path->socket_fd, &remote_is_big_endian, sizeof(remote_is_big_endian), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote_is_big_endian: %s\n", error_message);
    return false;
  }
  // Recv buffer count
  uint32_t buffer_count;
  if (!socketRecv(private_path->socket_fd, &buffer_count, sizeof(buffer_count), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv buffer_count: %s\n", error_message);
    return false;
  }
  if (remote_is_big_endian != is_big_endian) endianSwap4Byte(&buffer_count, 1);

  // Store the info
  private_path->remote_buffer_count = buffer_count;
  private_path->remote_buffers = NULL;
  if (buffer_count == 0) return true;

  // Allocate the remote buffer list
  private_path->remote_buffers = (RemoteTakyonBuffer *)calloc(buffer_count, sizeof(RemoteTakyonBuffer));
  if (private_path->remote_buffers == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }

  for (uint32_t i=0; i<buffer_count; i++) {
    // Recv bytes
    uint64_t bytes;
    if (!socketRecv(private_path->socket_fd, &bytes, sizeof(bytes), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv bytes: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap8Byte(&bytes, 1);
    // Recv remote_addr
    uint64_t remote_addr;
    if (!socketRecv(private_path->socket_fd, &remote_addr, sizeof(remote_addr), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote_addr: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap8Byte(&remote_addr, 1);
    // Recv remote_key
    uint32_t remote_key;
    if (!socketRecv(private_path->socket_fd, &remote_key, sizeof(remote_key), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote_key: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap4Byte(&remote_key, 1);

    // Store the results
    private_path->remote_buffers[i].bytes = bytes;
    private_path->remote_buffers[i].remote_addr = remote_addr;
    private_path->remote_buffers[i].remote_key = remote_key;
  }

  return true;
}

bool rdmaCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the provider
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(path->attrs.provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get provider name: %s\n", error_message);
    return false;
  }

  // Get all posible flags and values
  bool is_RC = (strcmp(provider_name, "RdmaRC") == 0);
  bool is_UC = (strcmp(provider_name, "RdmaUC") == 0);
  bool is_UD = (strcmp(provider_name, "RdmaUDUnicastSend") == 0 || strcmp(provider_name, "RdmaUDUnicastRecv") == 0);
  bool is_UD_sender = (strcmp(provider_name, "RdmaUDUnicastSend") == 0);
  bool is_client = argGetFlag(path->attrs.provider, "-client");
  bool is_server = argGetFlag(path->attrs.provider, "-server");
  bool allow_reuse = argGetFlag(path->attrs.provider, "-reuse");

  // -localIP=<ip_addr>|<hostname>|Any
  char local_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool local_ip_addr_found = false;
  bool ok = argGetText(path->attrs.provider, "-localIP=", local_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &local_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -localIP=<ip_addr>|<hostname>|Any is invalid: %s\n", error_message);
    return false;
  }

  // -remoteIP=<ip_addr>|<hostname>
  char remote_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool remote_ip_addr_found = false;
  ok = argGetText(path->attrs.provider, "-remoteIP=", remote_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &remote_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -remoteIP=<ip_addr>|<hostname> is invalid: %s\n", error_message);
    return false;
  }

  // -port=<port_number>
  uint32_t port_number = 0;
  bool port_number_found = false;
  ok = argGetUInt(path->attrs.provider, "-port=", &port_number, &port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -port=<port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (!port_number_found) {
    TAKYON_RECORD_ERROR(path->error_message, "Rdma needs the argument: -port=<port_number>\n");
    return false;
  }
  if (port_number_found) {
    if ((port_number < 1024) || (port_number > 65535)) {
      TAKYON_RECORD_ERROR(path->error_message, "port numbers need to be between 1024 and 65535\n");
      return false;
    }
  }

  // -rdmaDevice=<name>
  char rdma_device_name[TAKYON_MAX_PROVIDER_CHARS];
  bool rdma_device_name_found = false;
  ok = argGetText(path->attrs.provider, "-rdmaDevice=", rdma_device_name, TAKYON_MAX_PROVIDER_CHARS, &rdma_device_name_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -rdmaDevice=<name> is invalid: %s\n", error_message);
    return false;
  }
  if (!rdma_device_name_found) {
    TAKYON_RECORD_ERROR(path->error_message, "Rdma needs the argument: -rdmaDevice=<name>\n");
    return false;
  }

  // -rdmaPort=<local_rdma_port_number>
  uint32_t rdma_port_number = 0;
  bool rdma_port_number_found = false;
  ok = argGetUInt(path->attrs.provider, "-rdmaPort=", &rdma_port_number, &rdma_port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -rdmaPort=<local_rdma_port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (!rdma_port_number_found) {
    TAKYON_RECORD_ERROR(path->error_message, "Rdma needs the argument: -rdmaPort=<local_rdma_port_number>\n");
    return false;
  }
  if (rdma_port_number_found) {
    if (rdma_port_number == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "RDMA port needs to be greater than 0\n");
      return false;
    }
  }

  // [-mtuBytes=<n>]
  uint32_t mtu_bytes = 0;
  bool mtu_bytes_found = false;
  ok = argGetUInt(path->attrs.provider, "-mtuBytes=", &mtu_bytes, &mtu_bytes_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -mtuBytes=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (mtu_bytes_found && mtu_bytes != 256 && mtu_bytes != 512 && mtu_bytes != 1024 && mtu_bytes != 2048 && mtu_bytes != 4096) { 
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -mtuBytes=<n> must be one of 256, 512, 1024, 2048, or 4096\n");
    return false;
  }

  // [-gidIndex=<index>]
  uint32_t gid_index = 0;
  bool gid_index_found = false;
  ok = argGetUInt(path->attrs.provider, "-gidIndex=", &gid_index, &gid_index_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -gidIndex=<index> is invalid: %s\n", error_message);
    return false;
  }
  if (gid_index_found && gid_index >= 256) {
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -gidIndex=<n> must be less than 256\n");
    return false;
  }

  // [-serviceLevel=<n>]
  uint32_t service_level = 0;
  bool service_level_found = false;
  ok = argGetUInt(path->attrs.provider, "-serviceLevel=", &service_level, &service_level_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -serviceLevel=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (service_level_found && service_level >= 256) {
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -serviceLevel=<n> must be less than 256\n");
    return false;
  }

  // [-hopLimit=<n>]
  uint32_t hop_limit = 0;
  bool hop_limit_found = false;
  ok = argGetUInt(path->attrs.provider, "-hopLimit=", &hop_limit, &hop_limit_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -hopLimit=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (hop_limit_found && hop_limit >= 256) {
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -hopLimit=<n> must be less than 256\n");
    return false;
  }

  // [-retransmitTimeout=<n>]
  uint32_t retransmit_timeout = 0;
  bool retransmit_timeout_found = false;
  ok = argGetUInt(path->attrs.provider, "-retransmitTimeout=", &retransmit_timeout, &retransmit_timeout_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -retransmitTimeout=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (retransmit_timeout_found && retransmit_timeout > 31) { 
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -retransmitTimeout=<n> must be a value in the range [0 .. 31]\n");
    return false;
  }

  // [-retryCnt=<n>]
  uint32_t retry_cnt = 0;
  bool retry_cnt_found = false;
  ok = argGetUInt(path->attrs.provider, "-retryCnt=", &retry_cnt, &retry_cnt_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -retryCnt=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (retry_cnt_found && retry_cnt > 7) { 
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -retryCnt=<n> must be a value in the range [0 .. 7]\n");
    return false;
  }

  // [-rnrRetry=<n>]
  uint32_t rnr_retry = 0;
  bool rnr_retry_found = false;
  ok = argGetUInt(path->attrs.provider, "-rnrRetry=", &rnr_retry, &rnr_retry_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -rnrRetry=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (rnr_retry_found && rnr_retry > 7) { 
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -rnrRetry=<n> must be a value in the range [0 .. 7]\n");
    return false;
  }

  // [-minRnrTimer=<n>]
  uint32_t min_rnr_timer = 0;
  bool min_rnr_timer_found = false;
  ok = argGetUInt(path->attrs.provider, "-minRnrTimer=", &min_rnr_timer, &min_rnr_timer_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "attribute -minRnrTimer=<n> is invalid: %s\n", error_message);
    return false;
  }
  if (min_rnr_timer_found && min_rnr_timer > 31) { 
    TAKYON_RECORD_ERROR(path->error_message, "optional attribute -minRnrTimer=<n> must be a value in the range [0 .. 31]\n");
    return false;
  }

  // Validate arguments
  int num_modes = (is_RC ? 1 : 0) + (is_UC ? 1 : 0) + (is_UD ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "Rdma spec must start with one of RdmaRC, RdmaUC, RdmaUDUnicastSend, or RdmaUDUnicastRecv\n");
    return false;
  }
  num_modes = (is_client ? 1 : 0) + (is_server ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "Rdma must specify one of -client or -server\n");
    return false;
  }
  if (is_client) {
    if (!remote_ip_addr_found) {
      TAKYON_RECORD_ERROR(path->error_message, "Rdma needs the argument: -remoteIP=<ip_addr>|<hostname>\n");
      return false;
    }
  } else {
    if (!local_ip_addr_found) {
      TAKYON_RECORD_ERROR(path->error_message, "Rdma needs the argument: -localIP=<ip_addr>|<hostname>|Any\n");
      return false;
    }
  }

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;

  // Create the socket connection that will be used to handshake to RDMA connection
  if (is_client) {
    // Client
    if (!socketCreateTcpClient(remote_ip_addr, (uint16_t)port_number, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP client socket needed for the Rdma handshake: %s\n", error_message);
      goto failed;
    }
  } else {
    // Server
    if (!socketCreateTcpServer(local_ip_addr, (uint16_t)port_number, allow_reuse, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP server socket needed for the Rdma handshake: %s\n", error_message);
      goto failed;
    }
  }

  // RDMA send requests and SGEs
  uint32_t max_pending_read_and_atomic_requests = path->attrs.max_pending_read_requests + path->attrs.max_pending_atomic_requests;
  uint32_t max_pending_send_and_one_sided_requests = path->attrs.max_pending_send_requests + path->attrs.max_pending_write_requests + path->attrs.max_pending_read_requests + path->attrs.max_pending_atomic_requests;
  uint32_t max_sub_buffers_per_send_and_one_sided_request = MY_MAX(path->attrs.max_sub_buffers_per_send_request, path->attrs.max_sub_buffers_per_write_request);
  max_sub_buffers_per_send_and_one_sided_request = MY_MAX(max_sub_buffers_per_send_and_one_sided_request, path->attrs.max_sub_buffers_per_read_request);
  max_sub_buffers_per_send_and_one_sided_request = MY_MAX(max_sub_buffers_per_send_and_one_sided_request, 1);
  if (max_pending_send_and_one_sided_requests > 0) {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Max send requests=%d, sub buffers per request=%d\n", max_pending_send_and_one_sided_requests, max_sub_buffers_per_send_and_one_sided_request);
    private_path->rdma_send_request_list = (RdmaSendRequest *)calloc(max_pending_send_and_one_sided_requests, sizeof(RdmaSendRequest));
    if (private_path->rdma_send_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = max_pending_send_and_one_sided_requests * max_sub_buffers_per_send_and_one_sided_request;
    if (num_sges > 0) {
      private_path->send_sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->send_sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<max_pending_send_and_one_sided_requests; i++) {
        private_path->rdma_send_request_list[i].sges = &private_path->send_sge_list[i*max_sub_buffers_per_send_and_one_sided_request];
      }
    }
  }

  // RDMA recv requests and SGEs
  if (post_recv_count > 0 && path->attrs.max_pending_recv_requests == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "path->attrs.max_pending_recv_requests must be > 0 when post_recv_count > 0\n");
    goto failed;
  }
  if (path->attrs.max_pending_recv_requests > 0) {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Max recv requests=%d, sub buffers per request=%d\n", path->attrs.max_pending_recv_requests, path->attrs.max_sub_buffers_per_recv_request);
    private_path->rdma_recv_request_list = (RdmaRecvRequest *)calloc(path->attrs.max_pending_recv_requests, sizeof(RdmaRecvRequest));
    if (private_path->rdma_recv_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = path->attrs.max_pending_recv_requests * path->attrs.max_sub_buffers_per_recv_request;
    if (num_sges > 0) {
      private_path->recv_sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->recv_sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<path->attrs.max_pending_recv_requests; i++) {
        private_path->rdma_recv_request_list[i].sges = &private_path->recv_sge_list[i*path->attrs.max_sub_buffers_per_recv_request];
      }
    }
  }

  // Create the info need for RDMA buffer registrations
  private_path->rdma_buffer_list = calloc(path->attrs.buffer_count, sizeof(RdmaBuffer));
  if (private_path->rdma_buffer_list == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    goto failed;
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    private_path->rdma_buffer_list[i].path = path;   // Make sure each buffer knows it's for this path: need for verifications later on
    path->attrs.buffers[i].private_data = &private_path->rdma_buffer_list[i];
  }

  // Prepare for endpoint creation to post recvs
  for (uint32_t i=0; i<post_recv_count; i++) {
    RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_recv_request_index];
    private_path->curr_rdma_recv_request_index = (private_path->curr_rdma_recv_request_index + 1) % path->attrs.max_pending_recv_requests;
    recv_requests[i].private_data = rdma_request;
  }

  // Fill in the app options
  RdmaAppOptions app_options;
  app_options.mtu_bytes          = mtu_bytes_found ? mtu_bytes : 0;
  app_options.gid_index          = gid_index_found ? (uint8_t)gid_index : 0;
  app_options.service_level      = service_level_found ? (uint8_t)service_level : 0;
  app_options.hop_limit          = hop_limit_found ? (uint8_t)hop_limit : 1;
  app_options.retransmit_timeout = retransmit_timeout_found ? (uint8_t)retransmit_timeout : 14;
  app_options.retry_cnt          = retry_cnt_found ? (uint8_t)retry_cnt : 7;
  app_options.rnr_retry          = rnr_retry_found ? (uint8_t)rnr_retry : 6;
  app_options.min_rnr_timer      = min_rnr_timer_found ? (uint8_t)min_rnr_timer : 12;

  // Create the RDMA endpoint
  enum ibv_qp_type qp_type = (is_RC) ? IBV_QPT_RC : (is_UC) ? IBV_QPT_UC : IBV_QPT_UD;
  private_path->endpoint = rdmaCreateEndpoint(path, path->attrs.is_endpointA, private_path->socket_fd, qp_type, is_UD_sender, rdma_device_name, rdma_port_number,
					      max_pending_send_and_one_sided_requests, path->attrs.max_pending_recv_requests,
					      max_sub_buffers_per_send_and_one_sided_request, path->attrs.max_sub_buffers_per_recv_request,
                                              max_pending_read_and_atomic_requests,
					      post_recv_count, recv_requests, app_options, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (private_path->endpoint == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to create RDMA endpoint: %s\n", error_message);
    goto failed;
  }

  // Exchange buffer info
  if (path->attrs.is_endpointA) {
    if (!sendRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
    if (!recvRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
  } else {
    if (!recvRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
    if (!sendRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
  }

  // Create a pipe that will be used for disconnect detection
  if (!pipeCreate(&private_path->read_pipe_fd, &private_path->write_pipe_fd, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to create pipe used for disconnect detection: %s\n", error_message);
    goto failed;
  }

  // Start the thread to detect if the socket is disconnected
  int rc = pthread_create(&private_path->disconnect_detection_thread_id, NULL, disconnectDetectionThread, private_path);
  if (rc != 0) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start disconnectDetectionThread(): rc=%d\n", rc);
    goto failed;
  }
  private_path->thread_started = true;

  // Ready to start transferring
  return true;

 failed:
  if (private_path->read_pipe_fd != 0 && private_path->write_pipe_fd != 0) pipeDestroy(private_path->read_pipe_fd, private_path->write_pipe_fd);
  if (private_path->endpoint != NULL) (void)rdmaDestroyEndpoint(path, private_path->endpoint, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->send_sge_list != NULL) free(private_path->send_sge_list);
  if (private_path->recv_sge_list != NULL) free(private_path->recv_sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  if (private_path->remote_buffers != NULL) free(private_path->remote_buffers);
  free(private_path);
  return false;
}

bool rdmaDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  bool timed_out = false;

  // Provide some time for the remote side to get any in-transit data before disconnecting
  clockSleepYield(MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING);

  // Wake up thread
  if (private_path->thread_started) {
    if (!private_path->connection_failed) {
      // Wake thread up so it will exit
      uint32_t dummy = 0;
      if (!socketSend(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to wake up disconnectDetectionThread(): %s\n", error_message);
        private_path->connection_failed = true;
      }
    }
    // Wait for the thread to exit
    int rc = pthread_join(private_path->disconnect_detection_thread_id, NULL);
    if (rc != 0) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to join disconnectDetectionThread(): rc=%d\n", rc);
      private_path->connection_failed = true;
    }
  }

  // Socket barrier: to make sure pending transactions are complete
  bool barrier_ok = !private_path->connection_failed;
  if (barrier_ok) {
    if (path->attrs.is_endpointA) {
      uint32_t x = 23;
      // Send
      if (!socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barriar value: %s\n", error_message);
        return false;
      }
      x = 33;
      // Recv
      if (barrier_ok && !socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
        return false;
      }
      if (x != 23) {
        TAKYON_RECORD_ERROR(path->error_message, "Got incorrect barrier value: %s\n", error_message);
        return false;
      }
    } else {
      uint32_t x = 33;
      // Recv
      if (!socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
        return false;
      }
      // Send
      if (barrier_ok && !socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barrier value: %s\n", error_message);
        return false;
      }
    }
  } else {
    TAKYON_RECORD_ERROR(path->error_message, "This endpoint is trying to finalize, but the remote side seems to have disconnected\n");
    return false;
  }

  // Disconnect RDMA endpoint
  if (!rdmaDestroyEndpoint(path, endpoint, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to destroy RDMA endpoint: %s\n", error_message);
    return false;
  }

  // Free memory
  if (private_path->read_pipe_fd != 0 && private_path->write_pipe_fd != 0) pipeDestroy(private_path->read_pipe_fd, private_path->write_pipe_fd);
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->send_sge_list != NULL) free(private_path->send_sge_list);
  if (private_path->recv_sge_list != NULL) free(private_path->recv_sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  if (private_path->remote_buffers != NULL) free(private_path->remote_buffers);
  free(private_path);

  return true;
}

bool rdmaOneSided(TakyonPath *path, TakyonOneSidedRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds; // Quiet compiler
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // See if trying to read with UC
  if (!(request->operation == TAKYON_OP_WRITE || request->operation == TAKYON_OP_WRITE_WITH_PIGGYBACK) && endpoint->protocol != RDMA_PROTOCOL_RC) {
    TAKYON_RECORD_ERROR(path->error_message, "RDMA UC does not allow '%s'\n", takyonPrivateOneSidedOpToText(request->operation));
    return false;
  }

#ifdef EXTRA_ERROR_CHECKING
  // Make sure at least one buffer
  if (request->sub_buffer_count == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "One sided requests must have at least one sub buffer\n");
    return false;
  }
#endif

  // Get total bytes to transfer
  uint64_t total_local_bytes_to_transfer = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    uint64_t local_bytes = sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *local_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)local_buffer->private_data;
    if (rdma_buffer->path != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
    if (local_bytes > (local_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds local buffer (bytes = %ju)\n", local_bytes, sub_buffer->offset, local_buffer->bytes);
      return false;
    }
#endif
    total_local_bytes_to_transfer += local_bytes;
  }

  // Remote info
#ifdef EXTRA_ERROR_CHECKING
  if (request->remote_buffer_index >= private_path->remote_buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "Remote buffer index = %d is out of range\n", request->remote_buffer_index);
    return false;
  }
#endif
  RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[request->remote_buffer_index];
  uint64_t remote_addr = (uint64_t)remote_buffer->remote_addr + request->remote_offset;
  uint32_t remote_key = remote_buffer->remote_key;

  // Verify enough space in remote request
#ifdef EXTRA_ERROR_CHECKING
  uint64_t remote_max_bytes = remote_buffer->bytes - request->remote_offset;
  if (total_local_bytes_to_transfer > remote_max_bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough available remote bytes: bytes_to_transfer=%ju, remote_bytes=%ju remote_offset=%ju\n", total_local_bytes_to_transfer, remote_buffer->bytes, request->remote_offset);
    return false;
  }
#endif

  // Get the next unused rdma_request
  uint32_t max_pending_send_and_one_sided_requests = path->attrs.max_pending_send_requests + path->attrs.max_pending_write_requests + path->attrs.max_pending_read_requests + path->attrs.max_pending_atomic_requests;
  RdmaSendRequest *rdma_request = &private_path->rdma_send_request_list[private_path->curr_rdma_send_request_index];
  private_path->curr_rdma_send_request_index = (private_path->curr_rdma_send_request_index + 1) % max_pending_send_and_one_sided_requests;

  // Determine the operation
  enum ibv_wr_opcode transfer_mode;
  if (request->operation == TAKYON_OP_WRITE) transfer_mode = IBV_WR_RDMA_WRITE;
  else if (request->operation == TAKYON_OP_WRITE_WITH_PIGGYBACK) transfer_mode = IBV_WR_RDMA_WRITE_WITH_IMM;
  else if (request->operation == TAKYON_OP_READ) transfer_mode = IBV_WR_RDMA_READ;
  else if (request->operation == TAKYON_OP_ATOMIC_COMPARE_AND_SWAP_UINT64) transfer_mode = IBV_WR_ATOMIC_CMP_AND_SWP;
  else if (request->operation == TAKYON_OP_ATOMIC_ADD_UINT64) transfer_mode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  else {
    TAKYON_RECORD_ERROR(path->error_message, "One sided operation '%s' not supported\n", takyonPrivateOneSidedOpToText(request->operation));
    return false;
  }

  // Start the 'read/write' RDMA transfer
  uint64_t transfer_id = (uint64_t)request;
  struct ibv_sge *sge_list = rdma_request->sges;
  if (!rdmaEndpointStartSend(path, endpoint, transfer_mode, transfer_id, request->sub_buffer_count, request->sub_buffers, sge_list, request->atomics, remote_addr, remote_key, piggyback_message, request->submit_fence, request->use_is_done_notification, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start the RDMA '%s': %s\n", takyonPrivateOneSidedOpToText(request->operation), error_message);
    return false;
  }

  return true;
}

bool rdmaIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Determine the expected opcode
  enum ibv_wc_opcode expected_opcode;
  if (request->operation == TAKYON_OP_WRITE) expected_opcode = IBV_WC_RDMA_WRITE;
  else if (request->operation == TAKYON_OP_WRITE_WITH_PIGGYBACK) expected_opcode = IBV_WC_RDMA_WRITE;
  else if (request->operation == TAKYON_OP_READ) expected_opcode = IBV_WC_RDMA_READ;
  else if (request->operation == TAKYON_OP_ATOMIC_COMPARE_AND_SWAP_UINT64) expected_opcode = IBV_WC_COMP_SWAP;
  else if (request->operation == TAKYON_OP_ATOMIC_ADD_UINT64) expected_opcode = IBV_WC_FETCH_ADD;
  else {
    TAKYON_RECORD_ERROR(path->error_message, "One sided operation '%s' not supported\n", takyonPrivateOneSidedOpToText(request->operation));
    return false;
  }

  // See if the RDMA message is sent
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaEndpointIsSent(endpoint, expected_transfer_id, expected_opcode, private_path->read_pipe_fd, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for RDMA '%s' to complete: %s\n", takyonPrivateOneSidedOpToText(request->operation), error_message);
    return false;
  }

  return true;
}

bool rdmaSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds; // Quiet compiler
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

#ifdef EXTRA_ERROR_CHECKING
  // Validate source. Can't compare to dest, since it's unknown what buffers the message is going to
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)src_buffer->private_data;
    if (rdma_buffer->path != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
    uint64_t src_bytes = sub_buffer->bytes;
    if (src_bytes > (src_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds src buffer (bytes = %ju)\n", src_bytes, sub_buffer->offset, src_buffer->bytes);
      return false;
    }
  }
#endif

  // Get the next unused rdma_request
  uint32_t max_pending_send_and_one_sided_requests = path->attrs.max_pending_send_requests + path->attrs.max_pending_write_requests + path->attrs.max_pending_read_requests + path->attrs.max_pending_atomic_requests;
  RdmaSendRequest *rdma_request = &private_path->rdma_send_request_list[private_path->curr_rdma_send_request_index];
  private_path->curr_rdma_send_request_index = (private_path->curr_rdma_send_request_index + 1) % max_pending_send_and_one_sided_requests;

  // Start the 'send' RDMA transfer
  enum ibv_wr_opcode transfer_mode = IBV_WR_SEND_WITH_IMM;
  uint64_t transfer_id = (uint64_t)request;
  struct ibv_sge *sge_list = rdma_request->sges;
  uint64_t remote_addr = 0;
  uint32_t remote_key = 0;
  if (!rdmaEndpointStartSend(path, endpoint, transfer_mode, transfer_id, request->sub_buffer_count, request->sub_buffers, sge_list, NULL, remote_addr, remote_key, piggyback_message, request->submit_fence, request->use_is_sent_notification, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start the RDMA send: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret) {
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // See if the RDMA message is sent
  uint64_t expected_transfer_id = (uint64_t)request;
  enum ibv_wc_opcode expected_opcode = IBV_WC_SEND;
  if (!rdmaEndpointIsSent(endpoint, expected_transfer_id, expected_opcode, private_path->read_pipe_fd, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for RDMA send to complete: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Validate message attributes
#ifdef EXTRA_ERROR_CHECKING
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *request = &requests[i];
    for (uint32_t j=0; j<request->sub_buffer_count; j++) {
      TakyonSubBuffer *sub_buffer = &request->sub_buffers[j];
      if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
	TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
	return false;
      }
      TakyonBuffer *dest_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
      RdmaBuffer *rdma_buffer = (RdmaBuffer *)dest_buffer->private_data;
      if (rdma_buffer->path != path) {
        TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
        return false;
      }
      uint64_t dest_bytes = sub_buffer->bytes;
      if (dest_bytes > (dest_buffer->bytes - sub_buffer->offset)) {
	TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds recv buffer (bytes = %ju)\n", dest_bytes, sub_buffer->offset, dest_buffer->bytes);
	return false;
      }
    }
  }
#endif

  // Get the next block of unused rdma requests
  for (uint32_t i=0; i<request_count; i++) {
    RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_recv_request_index];
    private_path->curr_rdma_recv_request_index = (private_path->curr_rdma_recv_request_index + 1) % path->attrs.max_pending_recv_requests;
    requests[i].private_data = rdma_request;
  }

  // Post the recv requests
  if (!rdmaEndpointPostRecvs(path, endpoint, request_count, requests, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to post RDMA recv requests: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret) {
  *timed_out_ret = false;
  TakyonComm *comm = (TakyonComm *)path->private_data;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Wait for the message
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaEndpointIsRecved(endpoint, expected_transfer_id, private_path->read_pipe_fd, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS, bytes_received_ret, piggyback_message_ret)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv RDMA message: %s\n", error_message);
    return false;
  }

  return true;
}
