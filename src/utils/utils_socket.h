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
//   - Minor changes in this file
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

#ifndef _utils_socket_h_
#define _utils_socket_h_

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#include <winsock2.h>
#define TakyonSocket SOCKET // This is implicitly an unsigned int (which is why error checking is different)
#else
#define TakyonSocket int
#endif

#ifdef __cplusplus
extern "C"
{
#endif

// Connected (reliable) sockets
extern bool socketCreateLocalClient(const char *socket_name, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars);
extern bool socketCreateTcpClient(const char *ip_addr, uint16_t port_number, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars);
extern bool socketCreateLocalServer(const char *socket_name, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars);
extern bool socketCreateTcpServer(const char *ip_addr, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars);
#ifdef ENABLE_EPHEMERAL_PORT_MANAGER
extern bool socketCreateEphemeralTcpClient(const char *ip_addr, const char *provider_name, uint32_t path_id, TakyonSocket *socket_fd_ret, int64_t timeout_ns, uint64_t verbosity, char *error_message, int max_error_message_chars);
extern bool socketCreateEphemeralTcpServer(const char *ip_addr, const char *provider_name, uint32_t path_id, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars);
#endif
extern bool socketSend(TakyonSocket socket_fd, void *addr, size_t bytes_to_write, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars);
extern bool socketRecv(TakyonSocket socket_fd, void *data_ptr, size_t bytes_to_read, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars);

// Connectionless sockets (unreliable and may drop data)
extern bool socketCreateUnicastSender(const char *ip_addr, uint16_t port_number, TakyonSocket *socket_fd_ret, void **sock_in_addr_ret, char *error_message, int max_error_message_chars);
extern bool socketCreateUnicastReceiver(const char *ip_addr, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, char *error_message, int max_error_message_chars);
extern bool socketCreateMulticastSender(const char *ip_addr, const char *multicast_group, uint16_t port_number, bool disable_loopback, int ttl_level, TakyonSocket *socket_fd_ret, void **sock_in_addr_ret, char *error_message, int max_error_message_chars);
extern bool socketCreateMulticastReceiver(const char *ip_addr, const char *multicast_group, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, char *error_message, int max_error_message_chars);
extern bool socketDatagramSend(TakyonSocket socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars);
extern bool socketDatagramRecv(TakyonSocket socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars);

// Close socket
extern void socketClose(TakyonSocket socket_fd);

// Helpful socket functions
extern bool socketSetBlocking(TakyonSocket socket_fd, bool is_blocking, char *error_message, int max_error_message_chars);
extern bool socketSetKernelRecvBufferingSize(TakyonSocket socket_fd, int bytes, char *error_message, int max_error_message_chars); // Helpful to avoid dropping packets on receiver

#ifdef ENABLE_EPHEMERAL_PORT_MANAGER
// Ephemeral port manager (let OS find unused port numbers to be used by IP connections: sockets, RDMA)
extern void ephemeralPortManagerInit(uint64_t verbosity);
extern void ephemeralPortManagerFinalize();
extern void ephemeralPortManagerSet(const char *provider_name, uint32_t path_id, uint16_t ephemeral_port_number);
extern uint16_t ephemeralPortManagerGet(const char *provider_name, uint32_t path_id, int64_t timeout_ns, bool *timed_out_ret, uint64_t verbosity, char *error_message, int max_error_message_chars);
extern void ephemeralPortManagerRemove(const char *provider_name, uint32_t path_id, uint16_t ephemeral_port_number);
extern void ephemeralPortManagerRemoveLocally(const char *provider_name, uint32_t path_id);
#endif

// Pipes (helpful with closing sockets that are sleeping while waiting for activity)
extern bool pipeCreate(int *read_pipe_fd_ret, int *write_pipe_fd_ret, char *error_message, int max_error_message_chars);
extern bool socketWaitForDisconnectActivity(TakyonSocket socket_fd, int read_pipe_fd, bool *got_socket_activity_ret, char *error_message, int max_error_message_chars);
extern bool pipeWakeUpPollFunction(int write_pipe_fd, char *error_message, int max_error_message_chars);
extern void pipeDestroy(int read_pipe_fd, int write_pipe_fd);

#ifdef __cplusplus
}
#endif

#endif
