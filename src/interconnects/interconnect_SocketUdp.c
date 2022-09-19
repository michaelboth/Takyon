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

#include "interconnect_SocketUdp.h"
#include "takyon_private.h"
#include "utils_socket.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Supported formats:
//   UDP: No connection between endpoints (inter-process and inter-processor)
//     - Data packets can be dropped, arrive out of order, or duplicated
//     - Max transfer size is 64 KBs, but may be lower or limited to MTU size
//     - Useful where all bytes may not arrive (unreliable): e.g. live stream video or music
//   ---------------------------------------------------------------------------
//   Unicast: one sender, zero or one receiver. Transfers limited to 64 KBs
//     "SocketUdpSend -unicast -remoteIP=<ip_addr>|<hostname> -port=<port_number>"
//     "SocketUdpRecv -unicast -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse] [-rcvbuf=<bytes>]"
//   Multicast: one sender, zero to many receivers. Transfers typically limited to MTU bytes
//     "SocketUdpSend -multicast -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr> -port=<port_number> [-noLoopback] [-TTL=<time_to_live>]"
//     "SocketUdpRecv -multicast -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr> -port=<port_number> [-reuse] [-rcvbuf=<bytes>]"
//
//   Argument descriptions:
//     -port=<port_number> = [1024 .. 65535]
//     -rcvbuf=<bytes> is used to give the kernel more buffering to help the receiver avoid dropping packets
//     -groupIP=<multicast_ip_addr>: Valid multicast addresses: 224.0.0.0 through 239.255.255.255, but some are reserved
//     -TTL=<time_to_live>: Supported TTL values:
//           0:   Are restricted to the same host
//           1:   Are restricted to the same subnet
//           32:  Are restricted to the same site
//           64:  Are restricted to the same region
//           128: Are restricted to the same continent
//           255: Are unrestricted in scope

typedef struct {
  TakyonSocket socket_fd;
  void *sock_in_addr;
  bool socket_is_in_polling_mode; // By default, a socket is in blocking mode
  bool connection_failed;
  bool is_sender;
} PrivateTakyonPath;

bool udpSocketCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  (void)post_recv_count; // Quiet compiler checking
  (void)recv_requests; // Quiet compiler checking
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the interconnect
  char interconnect_name[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!argGetInterconnect(path->attrs.interconnect, interconnect_name, TAKYON_MAX_INTERCONNECT_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get interconnect name: %s\n", error_message);
    return false;
  }

  // Get all posible flags and values
  bool is_unicast = argGetFlag(path->attrs.interconnect, "-unicast");
  bool is_multicast = argGetFlag(path->attrs.interconnect, "-multicast");
  bool is_a_send = (strcmp(interconnect_name, "SocketUdpSend") == 0);
  bool is_a_recv = (strcmp(interconnect_name, "SocketUdpRecv") == 0);
  bool allow_reuse = argGetFlag(path->attrs.interconnect, "-reuse");
  bool disable_loopback = argGetFlag(path->attrs.interconnect, "-noLoopback");
  // -localIP=<ip_addr>|<hostname>|Any
  char local_ip_addr[TAKYON_MAX_INTERCONNECT_CHARS];
  bool local_ip_addr_found = false;
  bool ok = argGetText(path->attrs.interconnect, "-localIP=", local_ip_addr, TAKYON_MAX_INTERCONNECT_CHARS, &local_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -localIP=<ip_addr>|<hostname>|Any is invalid: %s\n", error_message);
    return false;
  }
  // -remoteIP=<ip_addr>|<hostname>
  char remote_ip_addr[TAKYON_MAX_INTERCONNECT_CHARS];
  bool remote_ip_addr_found = false;
  ok = argGetText(path->attrs.interconnect, "-remoteIP=", remote_ip_addr, TAKYON_MAX_INTERCONNECT_CHARS, &remote_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -remoteIP=<ip_addr>|<hostname> is invalid: %s\n");
    return false;
  }
  // -groupIP=<multicast_ip_addr>
  char group_ip_addr[TAKYON_MAX_INTERCONNECT_CHARS];
  bool group_ip_addr_found = false;
  ok = argGetText(path->attrs.interconnect, "-groupIP=", group_ip_addr, TAKYON_MAX_INTERCONNECT_CHARS, &group_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -groupIP=<multicast_ip_addr> is invalid: %s\n", error_message);
    return false;
  }
  if (group_ip_addr_found) {
    int tokens[4];
    int ntokens = sscanf(group_ip_addr, "%d.%d.%d.%d", &tokens[0], &tokens[1], &tokens[2], &tokens[3]);
    if (ntokens != 4 || tokens[0] < 224 || tokens[0] > 239 || tokens[1] < 0 || tokens[1] > 255 || tokens[2] < 0 || tokens[2] > 255 || tokens[3] < 0 || tokens[3] > 255) {
      TAKYON_RECORD_ERROR(path->error_message, "-groupIP=<multicast_ip_addr> must be in the range 224.0.0.0 through 239.255.255.255\n");
      return false;
    }
  }
  // -rcvbuf=<bytes>
  uint32_t recvbuf_bytes = 0;
  bool recvbuf_bytes_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-rcvbuf=", &recvbuf_bytes, &recvbuf_bytes_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -rcvbuf=<bytes> is invalid: %s\n", error_message);
    return false;
  }
  if (!recvbuf_bytes_found) {
    recvbuf_bytes = 0;
  }
  // -TTL=<time_to_live>
  uint32_t time_to_live = 1;
  bool time_to_live_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-TTL=", &time_to_live, &time_to_live_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -TTL=<time_to_live> is invalid: %s\n", error_message);
    return false;
  }
  if (time_to_live_found) {
    if (time_to_live > 255) {
      TAKYON_RECORD_ERROR(path->error_message, "-TTL values need to be between 0 and 255\n");
      return false;
    }
  } else {
    // Set the default to something safe
    time_to_live = 1;
  }
  // -port=<port_number>
  uint32_t port_number = 0;
  bool port_number_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-port=", &port_number, &port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect spec -port=<port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (port_number_found) {
    if ((port_number < 1024) || (port_number > 65535)) {
      TAKYON_RECORD_ERROR(path->error_message, "port numbers need to be between 1024 and 65535\n");
      return false;
    }
  }

  // Validate arguments
  int num_modes = (is_unicast ? 1 : 0) + (is_multicast ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "SocketUdp must specify one of -unicast or -multicast\n");
    return false;
  }
  num_modes = (is_a_send ? 1 : 0) + (is_a_recv ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "SocketUdp must ne one of SocketUdpSend or SocketUdpRecv\n");
    return false;
  }
  if (is_unicast && is_a_send && (!remote_ip_addr_found || !port_number_found)) {
    TAKYON_RECORD_ERROR(path->error_message, "-unicastSend needs the following arguments: -remoteIP=<ip_addr>|<hostname> -port=<port_number>\n");
    return false;
  } else if (is_unicast && is_a_recv && (!local_ip_addr_found || !port_number_found)) {
    TAKYON_RECORD_ERROR(path->error_message, "-unicastRecv needs the following arguments: -localIP=<ip_addr>|<hostname>|Any -port=<port_number>\n");
    return false;
  } else if (is_multicast && is_a_send && (!local_ip_addr_found || !group_ip_addr_found || !port_number_found)) {
    TAKYON_RECORD_ERROR(path->error_message, "-multicastSend needs the following arguments: -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr> -port=<port_number>\n");
    return false;
  } else if (is_multicast && is_a_recv && (!local_ip_addr_found || !group_ip_addr_found || !port_number_found)) {
    TAKYON_RECORD_ERROR(path->error_message, "-multicastRecv needs the following arguments: -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr> -port=<port_number>\n");
    return false;
  }

  // Make sure each buffer knows it's for this path: need for verifications later on
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    path->attrs.buffers[i].private = path;
  }

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;
  private_path->connection_failed = false;
  private_path->socket_fd = -1;
  private_path->sock_in_addr = NULL;
  private_path->socket_is_in_polling_mode = false;
  private_path->is_sender = is_a_send;

  // Create the one-sided socket
  if (is_unicast && is_a_send) {
    if (!socketCreateUnicastSender(remote_ip_addr, (uint16_t)port_number, &private_path->socket_fd, &private_path->sock_in_addr, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create unicast send socket: %s\n", error_message);
      free(private_path);
      return false;
    }
  } else if (is_unicast && is_a_recv) {
    if (!socketCreateUnicastReceiver(local_ip_addr, (uint16_t)port_number, allow_reuse, &private_path->socket_fd, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create unicast recv socket: %s\n", error_message);
      free(private_path);
      return false;
    }
    // Allow datagrams to be buffered in the OS to avoid dropping packets
    if (recvbuf_bytes > 0) {
      if (!socketSetKernelRecvBufferingSize(private_path->socket_fd, recvbuf_bytes, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to set socket's SO_RCVBUF to %d: %s\n", recvbuf_bytes, error_message);
        socketClose(private_path->socket_fd);
        free(private_path);
        return false;
      }
    }
  } else if (is_multicast && is_a_send) {
    if (!socketCreateMulticastSender(local_ip_addr, group_ip_addr, (uint16_t)port_number, disable_loopback, time_to_live, &private_path->socket_fd, &private_path->sock_in_addr, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create multicast send socket: %s\n", error_message);
      free(private_path);
      return false;
    }
  } else if (is_multicast && is_a_recv) {
    if (!socketCreateMulticastReceiver(local_ip_addr, group_ip_addr, (uint16_t)port_number, allow_reuse, &private_path->socket_fd, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create multicast recr socket: %s\n", error_message);
      free(private_path);
      return false;
    }
    // Allow datagrams to be buffered in the OS to avoid dropping packets
    if (recvbuf_bytes > 0) {
      if (!socketSetKernelRecvBufferingSize(private_path->socket_fd, recvbuf_bytes, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to set socket's SO_RCVBUF to %d: %s\n", recvbuf_bytes, error_message);
        socketClose(private_path->socket_fd);
        free(private_path);
        return false;
      }
    }
  }

  // Ready to start transferring
  return true;
}

bool udpSocketDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // NOTE: TCP_NODELAY is likely already active, so just need to provide some time for the remote side to get any in-transit data before a disconnect message is sent
  // NOTE: private_path->connection_failed may be true, but still want to provide time for remote side to handle arriving data
  clockSleepYield(MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING);

  // Disconnect
  socketClose(private_path->socket_fd);

  if (private_path->sock_in_addr != NULL) free(private_path->sock_in_addr);
  free(private_path);

  return true;
}

bool udpSocketSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret) {
  (void)piggy_back_message;
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Make sure sending is allowed
  if (!private_path->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This multicast/unicast endpoint can only be used for receiving.\n");
    private_path->connection_failed = true;
    return false;
  }

  // Make sure socket is in correct polling or event driven mode
  if (private_path->socket_is_in_polling_mode && request->use_polling_completion) {
    private_path->socket_is_in_polling_mode = request->use_polling_completion;
    if (!socketSetBlocking(private_path->socket_fd, !request->use_polling_completion, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Could not set the socket to '%s' mode: %s\n", request->use_polling_completion ? "polling" : "event driven", error_message);
      private_path->connection_failed = true;
      return false;
    }
  }

  // Get bytes and addr to send
  if (request->sub_buffer_count != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "Unicast/Multicast socket send only supports request->sub_buffer_count == 1\n");
    return false;
  }
  TakyonSubBuffer *sub_buffer = &request->sub_buffers[0];
  TakyonBuffer *src_buffer = sub_buffer->buffer;
  if (src_buffer->private != path) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[0] is not from this Takyon path\n");
    return false;
  }
  uint64_t src_bytes = sub_buffer->bytes;
  if (src_bytes > (src_buffer->bytes - sub_buffer->offset)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju exceeds src buffer\n", src_bytes);
    return false;
  }
  if (src_bytes == 0) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Message is zero bytes\n");
    return false;
  }
  void *src_addr = (void *)((uint64_t)src_buffer->addr + sub_buffer->offset);

  // Send the message
  if (!socketDatagramSend(private_path->socket_fd, private_path->sock_in_addr, src_addr, src_bytes, request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Failed to transfer data: %s\n", error_message);
    return false;
  }
  if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
    // Timed out but no data was transfered yet
    return true;
  }

  return true;
}

bool udpSocketIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Make sure sending is allowed
  if (private_path->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This multicast/unicast endpoint can only be used for sending.\n");
    private_path->connection_failed = true;
    return false;
  }

  // Make sure socket is in correct polling or event driven mode
  if (private_path->socket_is_in_polling_mode && request->use_polling_completion) {
    private_path->socket_is_in_polling_mode = request->use_polling_completion;
    if (!socketSetBlocking(private_path->socket_fd, !request->use_polling_completion, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Could not set the socket to '%s' mode: %s\n", request->use_polling_completion ? "polling" : "event driven", error_message);
      private_path->connection_failed = true;
      return false;
    }
  }

  // Get the recv max bytes and addr
  if (request->sub_buffer_count != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "Unicast/Multicast socket recv only supports request->sub_buffer_count == 1\n");
    return false;
  }
  TakyonSubBuffer *sub_buffer = &request->sub_buffers[0];
  TakyonBuffer *buffer = sub_buffer->buffer;
  if (buffer->private != path) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "'sub_buffers[0] is not from the remote Takyon path\n");
    return false;
  }
  uint64_t max_bytes = sub_buffer->bytes;
  if (max_bytes < (buffer->bytes - sub_buffer->offset)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju and exceeds buffer\n", max_bytes);
    return false;
  }
  if (max_bytes == 0) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Message is zero bytes\n");
    return false;
  }
  void *recv_addr = (void *)((uint64_t)buffer->addr + sub_buffer->offset);

  // Recv bytes
  uint64_t bytes_read = 0;
  if (!socketDatagramRecv(private_path->socket_fd, recv_addr, max_bytes, &bytes_read, request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Failed to receive data: %s\n", error_message);
    return false;
  }
  if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
    // Timed out but no data was transfered yet
    return true;
  }

  // Return results
  *bytes_received_ret = bytes_read;
  *piggy_back_message_ret = 0; // Not supported

  return true;
}
