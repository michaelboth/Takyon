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


#include "utils_socket.h"
#include "takyon.h"   // Need for TAKYON_MAX_PROVIDER_CHARS and TAKYON_VERBOSITY_*
#include "utils_socket.h"
#include "utils_endian.h"
#include "utils_time.h"
#include "utils_thread_cond_timed_wait.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VXWORKS_7
#else
// DANGEROUS: If USE_AT_EXIT_METHOD is not used, the alternative is not thread safe, so all path's should be created/destroyed from the same thread to avoid race conditions
#define USE_AT_EXIT_METHOD
#endif

// If Takyon becomes a standard, would be nice to see the internet consortium reserve TAKYON_MULTICAST_PORT and TAKYON_MULTICAST_GROUP
#define TAKYON_MULTICAST_IP    "127.0.0.1"    // A local interface that is multicast capable (for both sending and receiving)
#define TAKYON_MULTICAST_PORT  6736           // Uses phone digits to spell "Open"
#define TAKYON_MULTICAST_GROUP "229.82.29.66" // Uses phone digits to spell "Takyon" i.e. 229.TA.KY.ON
#define TAKYON_MULTICAST_TTL   1              // Restrict to same subnet

//#define DEBUG_MESSAGE
//#define WARNING_MESSAGE

#define REQUEST_TIMEOUT_NS 1000000 // 1 millisecond
#define COND_WAIT_TIMEOUT_NS 1000000000 // 1 second
#define CHECK_FOR_EXIT_TIMEOUT_NS (1000000000/4) // 1/4 second

#define MAX_PRIVATE_ERROR_MESSAGE_CHARS 10000

enum {
  NEW_EPHEMERAL_PORT = 33,
  REQUEST_EPHEMERAL_PORT,
  EPHEMERAL_PORT_CONNECTED
};

typedef struct {
  bool in_use;
  uint32_t path_id;
  uint16_t ephemeral_port_number;
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
} EphemeralPortManagerItem;

#pragma pack(push, 1)  // Make sure no gaps in data structure
typedef struct {
  unsigned char command;
  unsigned char is_big_endian;
  uint16_t ephemeral_port_number;
  uint32_t path_id;
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
} EphemeralPortMessage;
#pragma pack(pop)

#ifdef USE_AT_EXIT_METHOD
static pthread_once_t L_once_control = PTHREAD_ONCE_INIT;
static pthread_mutex_t *L_mutex = NULL;
static pthread_cond_t *L_cond = NULL;
#else
static uint32_t L_usage_counter = 0;
static pthread_mutex_t L_mutex_private = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t L_cond_private = PTHREAD_COND_INITIALIZER;
#define L_mutex (&L_mutex_private)
#define L_cond (&L_cond_private)
#endif
static EphemeralPortManagerItem *L_manager_items = NULL;
static uint32_t L_num_manager_items = 0;
static char L_error_message[MAX_PRIVATE_ERROR_MESSAGE_CHARS]; // This manager is a global resource not always connected to a path, so need a separate error_message buffer
static TakyonSocket L_multicast_send_socket;
static void *L_multicast_send_socket_in_addr = NULL;
static TakyonSocket L_multicast_recv_socket;
#ifdef _WIN32
static bool L_thread_running = false;
#else
static int L_read_pipe_fd;
static int L_write_pipe_fd;
#endif
static pthread_t L_thread_id;
static uint64_t L_verbosity = TAKYON_VERBOSITY_NONE;

static void addItem(const char *provider_name, uint32_t path_id, uint16_t ephemeral_port_number) {
  // Check if already in list
  for (uint32_t i=0; i<L_num_manager_items; i++) {
    EphemeralPortManagerItem *item = &L_manager_items[i];
    if (item->in_use && item->path_id == path_id && strcmp(item->provider_name, provider_name)==0) {
      if (item->ephemeral_port_number != ephemeral_port_number) {
        // A path with the provider and ID might have been shut down and restarted
        // Since this is creating a port number, overwrite the potentially stale port number
        item->ephemeral_port_number = ephemeral_port_number;
      }
      // Already in list
      return;
    }
  }

  if (L_verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Ephemeral port manager: adding item (provider='%s', path_id=%u, ephemeral_port_number=%hu)\n", provider_name, path_id, ephemeral_port_number);
  }

  // See if there is an unused item
  EphemeralPortManagerItem *item = NULL;
  for (uint32_t i=0; i<L_num_manager_items; i++) {
    if (!L_manager_items[i].in_use) {
      // Found an empty item
      item = &L_manager_items[i];
      break;
    }
  }

  // Increase the size of the list if it's full
  if (item == NULL) {
    L_manager_items = realloc(L_manager_items, 2*L_num_manager_items*sizeof(EphemeralPortManagerItem));
    if (L_manager_items == NULL) {
      fprintf(stderr, "Out of memory\n");
      exit(EXIT_FAILURE);
    }
    for (uint32_t i=L_num_manager_items; i<L_num_manager_items*2; i++) {
      L_manager_items[i].in_use = false;
    }
    item = &L_manager_items[L_num_manager_items];
    L_num_manager_items *= 2;
  }

  // Add new item
  item->in_use = true;
  item->path_id = path_id;
  item->ephemeral_port_number = ephemeral_port_number;
  strncpy(item->provider_name, provider_name, TAKYON_MAX_PROVIDER_CHARS-1);
  item->provider_name[TAKYON_MAX_PROVIDER_CHARS-1] = '\0';
}

static bool hasItem(const char *provider_name, uint32_t path_id, uint16_t *ephemeral_port_number_ret) {
  for (uint32_t i=0; i<L_num_manager_items; i++) {
    EphemeralPortManagerItem *item = &L_manager_items[i];
    if (item->in_use && item->path_id == path_id && strcmp(item->provider_name, provider_name)==0) {
      *ephemeral_port_number_ret = item->ephemeral_port_number;
      return true;
    }
  }
  return false;
}

static void removeItem(const char *provider_name, uint32_t path_id) {
#ifdef DEBUG_MESSAGE
  printf("Ephemeral port manager: removing item (provider='%s', path_id=%u)\n", provider_name, path_id);
#endif
  for (uint32_t i=0; i<L_num_manager_items; i++) {
    EphemeralPortManagerItem *item = &L_manager_items[i];
    if (item->in_use) {
      if (item->path_id == path_id && strcmp(item->provider_name, provider_name)==0) {
        item->in_use = false;
      }
    }
  }
}

static void *ephemeralPortMonitoringThread(void *user_arg) {
  (void)user_arg; // Quiet the compiler checking
  bool this_is_big_endian = endianIsBig();

  while (true) {
#ifdef _WIN32
    // Wait for a multicast message to arrive
    int64_t timeout_ns = CHECK_FOR_EXIT_TIMEOUT_NS;
    bool is_polling = false;
    uint64_t bytes_read;
    EphemeralPortMessage message;
    bool timed_out = false;
#ifdef DEBUG_MESSAGE
    printf("Ephemeral port manager: Waiting for multicast message\n");
#endif
    if (!socketDatagramRecv(L_multicast_recv_socket, &message, sizeof(EphemeralPortMessage), &bytes_read, is_polling, timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
      // Failed to read the message
      fprintf(stderr, "Ephemeral port manager thread: Failed to read pending datagram: %s\n", L_error_message);
      exit(EXIT_FAILURE);
    }
    if (!L_thread_running) {
      // Time to gracefully exit
      break;
    }
    if (timed_out) {
      // No message so try again
      continue;
    }
#ifdef DEBUG_MESSAGE
    printf("Ephemeral port manager: Got multicast message\n");
#endif
#else

    // Wait for a multicast message to arrive
#ifdef DEBUG_MESSAGE
    printf("Ephemeral port manager: Waiting for multicast message\n");
#endif
    bool got_socket_activity;
    bool ok = socketWaitForDisconnectActivity(L_multicast_recv_socket, L_read_pipe_fd, &got_socket_activity, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS);
    if (!ok) {
      fprintf(stderr, "Ephemeral port manager thread: Failed to wait for activity on the multicast receive socket: %s\n", L_error_message);
      exit(EXIT_FAILURE);
    }
    if (!got_socket_activity) {
      // The read pipe has woken up this thread
      // Time to gracefully exit
      break;
    }

    // A multicast message arrived, so read it
#ifdef DEBUG_MESSAGE
    printf("Ephemeral port manager: Reading multicast message\n");
#endif
    int64_t timeout_ns = TAKYON_WAIT_FOREVER;
    bool is_polling = false;
    uint64_t bytes_read;
    EphemeralPortMessage message;
    bool timed_out = false;
    if (!socketDatagramRecv(L_multicast_recv_socket, &message, sizeof(EphemeralPortMessage), &bytes_read, is_polling, timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
      // Failed to read the message
      fprintf(stderr, "Ephemeral port manager thread: Failed to read pending datagram: %s\n", L_error_message);
      exit(EXIT_FAILURE);
    }
#endif

    // Interpret the message
    if (message.command != NEW_EPHEMERAL_PORT && message.command != REQUEST_EPHEMERAL_PORT && message.command != EPHEMERAL_PORT_CONNECTED) {
      fprintf(stderr, "Ephemeral port manager thread: Got invalid command = %d\n", message.command);
      exit(EXIT_FAILURE);
    }
    if (this_is_big_endian != message.is_big_endian) {
      // Endian swap the port number and path ID
      endianSwap2Byte(&message.ephemeral_port_number, 1);
      endianSwap4Byte(&message.path_id, 1);
    }
    if (message.command == NEW_EPHEMERAL_PORT) {
      // Record the info and wake up any threads waiting for it
      pthread_mutex_lock(L_mutex);
      addItem(message.provider_name, message.path_id, message.ephemeral_port_number);
      pthread_mutex_unlock(L_mutex);
      // Wake up all threads waiting on an ephemeral port number
      pthread_cond_broadcast(L_cond);
    } else if (message.command == REQUEST_EPHEMERAL_PORT) {
      pthread_mutex_lock(L_mutex);
      if (hasItem(message.provider_name, message.path_id, &message.ephemeral_port_number)) {
        // Exists in database, so re-multicast it
#ifdef DEBUG_MESSAGE
        printf("Ephemeral port manager: Re-multicast (provider='%s', path_id=%u, ephemeral_port_number=%hu)\n", message.provider_name, message.path_id, message.ephemeral_port_number);
#endif
        message.command = NEW_EPHEMERAL_PORT;
        message.is_big_endian = this_is_big_endian;
        // NOTE: this send is within a mutex, but this port manager is not used once a coonection is made and will not perturb communication performance
        int64_t heartbeat_timeout_ns = REQUEST_TIMEOUT_NS; // Need some time to get the message out, so ignore the path's timeout period
        bool timed_out = false;
        if (!socketDatagramSend(L_multicast_send_socket, L_multicast_send_socket_in_addr, &message, sizeof(EphemeralPortMessage), is_polling, heartbeat_timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
#ifdef WARNING_MESSAGE
          fprintf(stderr, "Warning: Ephemeral port manager thread: Failed to re-multicast ephemeral port info: %s. Will eventually retry\n", L_error_message);
#endif          
        }
        // NOTE: if timed out, just ignore since a subsequent attempt will be done if needed
      }
      pthread_mutex_unlock(L_mutex);
    } else if (message.command == EPHEMERAL_PORT_CONNECTED) {
      // Remove item from the list: this message will go to all managers on the sub net, but no gaurantee it wont get dropped. Still need to locally remove at endpoints
      // NOTE: This will be called by the endpoint that received the port number
      pthread_mutex_lock(L_mutex);
      removeItem(message.provider_name, message.path_id);
      pthread_mutex_unlock(L_mutex);
    }
  }

  // Time to exit
#ifdef DEBUG_MESSAGE
  printf("Ephemeral port manager: time to exit\n");
#endif
  return NULL;
}

static void ephemeralPortManagerFinalizePrivate(void) {
  if (L_verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Finalizing the ephemeral port manager\n");
  }

  // Wake up the thread
#ifdef _WIN32
  pthread_mutex_lock(L_mutex);
  L_thread_running = false;
  pthread_mutex_unlock(L_mutex);
#else
  // Wake up thread by signalling the pipe
  if (!pipeWakeUpPollFunction(L_write_pipe_fd, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
    // Failed to read the message
    fprintf(stderr, "Failed to use pipe to wake up ephemeral port manager thread: %s\n", L_error_message);
    exit(EXIT_FAILURE);
  }
#endif

  // Wake for thread to exit
  if (pthread_join(L_thread_id, NULL) != 0) {
    fprintf(stderr, "Failed to wait for the ephemeral port manager thread to exit\n");
    exit(EXIT_FAILURE);
  }

  // Free all pipes
#ifdef _WIN32
  // Nothing to do
#else
  pipeDestroy(L_read_pipe_fd, L_write_pipe_fd);
#endif

  // Close the sockets
  socketClose(L_multicast_recv_socket);
  socketClose(L_multicast_send_socket);

  // Free the manager items
  free(L_manager_items);
  L_manager_items = NULL;
  L_num_manager_items = 0;

#ifdef USE_AT_EXIT_METHOD
  // Free the mutex
  pthread_mutex_destroy(L_mutex);
  free(L_mutex);
  L_mutex = NULL;

  // Free the cond var
  pthread_cond_destroy(L_cond);
  free(L_cond);
  L_cond = NULL;
#endif

  if (L_verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Done finalizing the ephemeral port manager\n");
  }
}

static void ephemeralPortManagerInitOnce(void) {
  if (L_verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Initializing the ephemeral port manager\n");
  }

  // Get the defaults for setting up multicasting
  const char *multicast_ip    = TAKYON_MULTICAST_IP;
  uint16_t multicast_port     = TAKYON_MULTICAST_PORT;
  const char *multicast_group = TAKYON_MULTICAST_GROUP;
  int multicast_ttl           = TAKYON_MULTICAST_TTL;

  // Check to see if any of the defualts have been overridden
  const char *user_defined_multicast_ip = getenv("TAKYON_MULTICAST_IP");
  if (user_defined_multicast_ip != NULL) {
    multicast_ip = user_defined_multicast_ip;
  }
  const char *user_defined_multicast_port = getenv("TAKYON_MULTICAST_PORT");
  if (user_defined_multicast_port != NULL) {
    uint16_t temp_value;
    int count = sscanf(user_defined_multicast_port, "%hu", &temp_value);
    if (count == 1 && temp_value >= 1024 /* && temp_value <= 65535 */) {
      multicast_port = temp_value;
    } else {
      fprintf(stderr, "Warning: ignoring env TAKYON_MULTICAST_PORT='%s', this is not an unsigned short integer between 1024 and 65535 inclusive\n", user_defined_multicast_port);
    }
  }
  const char *user_defined_multicast_group = getenv("TAKYON_MULTICAST_GROUP");
  if (user_defined_multicast_group != NULL) {
    multicast_group = user_defined_multicast_group;
  }
  const char *user_defined_multicast_ttl = getenv("TAKYON_MULTICAST_TTL");
  if (user_defined_multicast_ttl != NULL) {
    uint16_t temp_value;
    int count = sscanf(user_defined_multicast_ttl, "%hu", &temp_value);
    if (count == 1 && temp_value <= 255) {
      multicast_ttl = temp_value;
    } else {
      fprintf(stderr, "Warning: ignoring env TAKYON_MULTICAST_TTL='%s', this is not an unsigned integer between 0 and 255\n", user_defined_multicast_ttl);
    }
  }

  if (L_verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("  TAKYON_MULTICAST_IP    = %s\n", multicast_ip);
    printf("  TAKYON_MULTICAST_PORT  = %hu\n", multicast_port);
    printf("  TAKYON_MULTICAST_GROUP = %s\n", multicast_group);
    printf("  TAKYON_MULTICAST_TTL   = %d\n", multicast_ttl);
  }

  // Start the resource allocations
  L_num_manager_items = 1;
  L_manager_items = calloc(L_num_manager_items, sizeof(EphemeralPortManagerItem));
  if (L_manager_items == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(EXIT_FAILURE);
  }

#ifdef USE_AT_EXIT_METHOD
  L_mutex = calloc(1, sizeof(pthread_mutex_t));
  if (L_mutex == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(EXIT_FAILURE);
  }
  pthread_mutex_init(L_mutex, NULL);

  L_cond = calloc(1, sizeof(pthread_cond_t));
  if (L_cond == NULL) {
    fprintf(stderr, "Out of memory\n");
    exit(EXIT_FAILURE);
  }
  pthread_cond_init(L_cond, NULL);
#endif

  // Create the multicast sender socket
  L_error_message[0] = '\0';
  bool disable_loopback = false;
  if (!socketCreateMulticastSender(multicast_ip, multicast_group, multicast_port, disable_loopback, multicast_ttl, &L_multicast_send_socket, &L_multicast_send_socket_in_addr, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
    fprintf(stderr, "Failed to create multicast sender for managing ephemeral port numbers: %s\n", L_error_message);
    exit(EXIT_FAILURE);
  }

  // Create the multicast receiver socket
  bool allow_reuse = true; // Need to allow other processes in the same OS to use the same port number
  /*+ this fails on OSX because port number already in use (probably by above socket sender, may need two port numbers?) */
  if (!socketCreateMulticastReceiver(multicast_ip, multicast_group, multicast_port, allow_reuse, &L_multicast_recv_socket, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
    fprintf(stderr, "Failed to create multicast receiver for managing ephemeral port numbers: %s\n", L_error_message);
    exit(EXIT_FAILURE);
  }

#ifdef _WIN32
  L_thread_running = true;
#else
  // Create pipe needed to shut down monitoring thread
  if (!pipeCreate(&L_read_pipe_fd, &L_write_pipe_fd, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
    fprintf(stderr, "Failed to create pipe needed to manage the life of the ephemeral port manager thread: %s\n", L_error_message);
    exit(EXIT_FAILURE);
  }
#endif

  // Start multicast recv monitoring thread
  // IMPORTANT: the thread now owns L_error_message
  if (pthread_create(&L_thread_id, NULL, ephemeralPortMonitoringThread, NULL) != 0) {
    fprintf(stderr, "Failed to start the ephemeral port number monitoring thread\n");
    exit(EXIT_FAILURE);
  }

#ifdef USE_AT_EXIT_METHOD
  // This will get called if the app calls exit() or if main does a normal return.
  if (atexit(ephemeralPortManagerFinalizePrivate) == -1) {
    fprintf(stderr, "Failed to setup atexit() in the ephemeral port manager initializer\n");
    exit(EXIT_FAILURE);
  }
#endif
}

// NOTE: Called once to init the manager thread
void ephemeralPortManagerInit(uint64_t verbosity) {
  L_verbosity = verbosity;
#ifdef USE_AT_EXIT_METHOD
  // Call this to make sure the ephemeral port manager is ready to coordinate: This can be called multiple times, but it's guaranteed to atomically run only the first time called.
  if (pthread_once(&L_once_control, ephemeralPortManagerInitOnce) != 0) {
    fprintf(stderr, "Failed to start the ephemeral port manager\n");
    exit(EXIT_FAILURE);
  }
#else
  // Increase usage counter
  pthread_mutex_lock(L_mutex);
  if (L_usage_counter == 0) {
    ephemeralPortManagerInitOnce();
  }
  L_usage_counter++;
  pthread_mutex_unlock(L_mutex);
#endif
}

void ephemeralPortManagerFinalize() {
#ifdef USE_AT_EXIT_METHOD
  // Nothing to do
#else
  // Decrease usage counter
  pthread_mutex_lock(L_mutex);
  if (L_usage_counter == 0) {
    fprintf(stderr, "ephemeralPortManagerFinalize() was called more times than ephemeralPortManagerInit(). This should never happen.\n");
    pthread_mutex_unlock(L_mutex);
    exit(EXIT_FAILURE);
  }
  L_usage_counter--;
  bool need_to_finalize = false;
  if (L_usage_counter == 0) {
    need_to_finalize = true;
  }
  pthread_mutex_unlock(L_mutex);
  if (need_to_finalize) {
    // DANGEROUS: This is not thread safe, so all path's should be created/destroyed from the same thread to avoid race conditions
    ephemeralPortManagerFinalizePrivate();
  }
#endif
}

// NOTE: This is called by the endpoint that wants a port number
uint16_t ephemeralPortManagerGet(const char *provider_name, uint32_t path_id, int64_t timeout_ns, bool *timed_out_ret, uint64_t verbosity, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Ephemeral port manager: asking for port for (provider='%s', path_id=%u)\n", provider_name, path_id);
  }

  int64_t start_time = clockTimeNanoseconds();

  // Keep trying until the timeout period occurs
  uint16_t ephemeral_port_number = 0;
  pthread_mutex_lock(L_mutex);
  while (!hasItem(provider_name, path_id, &ephemeral_port_number)) {
    // Get time remaining
    // NOTE: Don't want cond wait to wait the full timeout (e.g. wait forever) to allow for periodic heartbeats to multcast a request for the ephemeral port number
    int64_t cond_wait_timeout = COND_WAIT_TIMEOUT_NS;
    if (timeout_ns >= 0) {
      int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
      if (ellapsed_time_ns >= timeout_ns) {
        // Timed out
        pthread_mutex_unlock(L_mutex);
        *timed_out_ret = true;
        return 0;
      }
      int64_t remaining_timeout = timeout_ns - ellapsed_time_ns;
      if (cond_wait_timeout > remaining_timeout) {
        cond_wait_timeout = remaining_timeout;
      }
    }

    // Send muticast message to ask for port number
    {
#ifdef DEBUG_MESSAGE
      printf("Ephemeral port manager: asking again for port for (provider='%s', path_id=%u)\n", provider_name, path_id);
#endif
      EphemeralPortMessage message;
      message.command = REQUEST_EPHEMERAL_PORT;
      message.is_big_endian = endianIsBig();
      message.ephemeral_port_number = 0;
      message.path_id = path_id;
      strncpy(message.provider_name, provider_name, TAKYON_MAX_PROVIDER_CHARS-1);
      message.provider_name[TAKYON_MAX_PROVIDER_CHARS-1] = '\0';
      bool is_polling = false;
      int64_t heartbeat_timeout_ns = REQUEST_TIMEOUT_NS; // Need some time to get the message out, so ignore the path's timeout period
      bool timed_out = false;
      // NOTE: this send is within a mutex, but this port manager is not used once a coonection is made and will not perturb communication performance
      if (!socketDatagramSend(L_multicast_send_socket, L_multicast_send_socket_in_addr, &message, sizeof(EphemeralPortMessage), is_polling, heartbeat_timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
#ifdef WARNING_MESSAGE
        fprintf(stderr, "Warning: Ephemeral port manager: Failed to re-multicast ephemeral port info: %s. Will eventually retry.\n", L_error_message);
#endif
      }
      // NOTE: if timed out, just ignore since a subsequent attempt will be done if needed
    }

    // Sleep while waiting for data
    bool timed_out = false;
    bool suceeded = threadCondWait(L_mutex, L_cond, cond_wait_timeout, &timed_out, error_message, max_error_message_chars);
    if (!suceeded) {
      pthread_mutex_unlock(L_mutex);
      // Error message already set
      return 0;
    }
    // NOTE: ignore if this cond wait timed out, since the begining of the while loop will check
  }

  // Remove from the database so if it's stale, it can be later replaced with the correct one
  removeItem(provider_name, path_id);

  pthread_mutex_unlock(L_mutex);

  if (verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("Ephemeral port manager: got port for (provider='%s', path_id=%u) ephemeral_port_number=%hu\n", provider_name, path_id, ephemeral_port_number);
  }

  return ephemeral_port_number;
}

// NOTE: This will be call by the creator of the port number
void ephemeralPortManagerRemoveLocally(const char *provider_name, uint32_t path_id) {
  // Remove the item locally since the socket creation is done with trying to create (succefully or not)
  pthread_mutex_lock(L_mutex);
  removeItem(provider_name, path_id);
  pthread_mutex_unlock(L_mutex);
}

// NOTE: This is called by the receiver of the port number after the connection succesfully is made
void ephemeralPortManagerRemove(const char *provider_name, uint32_t path_id, uint16_t ephemeral_port_number) {
  pthread_mutex_lock(L_mutex);

  // Remove the item locally in case the datagram is dropped or no loopback
  removeItem(provider_name, path_id);

  EphemeralPortMessage message;
  message.command = EPHEMERAL_PORT_CONNECTED;
  message.is_big_endian = endianIsBig();
  message.ephemeral_port_number = ephemeral_port_number;
  message.path_id = path_id;
  strncpy(message.provider_name, provider_name, TAKYON_MAX_PROVIDER_CHARS-1);
  message.provider_name[TAKYON_MAX_PROVIDER_CHARS-1] = '\0';
  bool is_polling = false;
  int64_t heartbeat_timeout_ns = REQUEST_TIMEOUT_NS; // Need some time to get the message out, so ignore the path's timeout period
  // NOTE: this send is within a mutex, but this port manager is not used once a coonection is made and will not perturb communication performance
  bool timed_out = false;
  if (!socketDatagramSend(L_multicast_send_socket, L_multicast_send_socket_in_addr, &message, sizeof(EphemeralPortMessage), is_polling, heartbeat_timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
#ifdef WARNING_MESSAGE
    fprintf(stderr, "Warning: Ephemeral port manager: Failed to multicast EPHEMERAL_PORT_CONNECTED command: %s\n", L_error_message);
#endif
    // Should be benign since the entry in the database will eventually expire
  }

  pthread_mutex_unlock(L_mutex);
}

// NOTE: This is called by the creator of the ephemeral port number
void ephemeralPortManagerSet(const char *provider_name, uint32_t path_id, uint16_t ephemeral_port_number) {
  pthread_mutex_lock(L_mutex);

  // IMPORTANT: Need to guarantee that it gets into this local database in case the multicast message gets dropped
  EphemeralPortMessage message;
  message.command = NEW_EPHEMERAL_PORT;
  message.is_big_endian = endianIsBig();
  message.ephemeral_port_number = ephemeral_port_number;
  message.path_id = path_id;
  strncpy(message.provider_name, provider_name, TAKYON_MAX_PROVIDER_CHARS-1);
  message.provider_name[TAKYON_MAX_PROVIDER_CHARS-1] = '\0';
  bool is_polling = false;
  int64_t heartbeat_timeout_ns = REQUEST_TIMEOUT_NS; // Need some time to get the message out
  // NOTE: this send is within a mutex, but this port manager is not used once a coonection is made and will not perturb communication performance
  bool timed_out = false;
  if (!socketDatagramSend(L_multicast_send_socket, L_multicast_send_socket_in_addr, &message, sizeof(EphemeralPortMessage), is_polling, heartbeat_timeout_ns, &timed_out, L_error_message, MAX_PRIVATE_ERROR_MESSAGE_CHARS)) {
#ifdef WARNING_MESSAGE
    fprintf(stderr, "Warning: Ephemeral port manager: Failed to multicast NEW_EPHEMERAL_PORT command: %s\n", L_error_message);
#endif
    // Not a big deal since the port is in the database and a remote request will eventually get it to broadcast
  }

  pthread_mutex_unlock(L_mutex);
}
