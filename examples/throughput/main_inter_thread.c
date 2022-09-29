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

#include "throughput.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

static const char *L_provider = NULL;
static uint32_t L_iterations = 1000000;
static uint64_t L_message_bytes = 1024;
static uint32_t L_send_buffer_count = 10;
static uint32_t L_recv_buffer_count = 10;
static bool L_use_polling_completion = true;
static bool L_two_sided = true;
static bool L_validate = false;

static void *throughputThread(void *user_data) {
  bool is_endpointA = (user_data != NULL);
  throughput(is_endpointA, L_provider, L_iterations, L_message_bytes, L_send_buffer_count, L_recv_buffer_count, L_use_polling_completion, L_two_sided, L_validate);
  return NULL;
}

static void printUsageAndExit(const char *program) {
  printf("usage: %s \"<provider>\" [-h] [-n=<uint32>] [-b=<uint64>] [-s=<uint32>] [-r=<uint32>] [-e] [-o] [-v]\n", program);
  printf("   -h          : Print this message and exit\n");
  printf("   -n=<uint32> : Number of messages to send. Default is %u\n", L_iterations);
  printf("   -b=<uint64> : Bytes per message. Can use 0 if two-sided and supported by provider. Default is " UINT64_FORMAT "\n", L_message_bytes);
  printf("   -s=<uint32> : Send/read/write message buffer count. Default is %u\n", L_send_buffer_count);
  printf("   -r=<uint32> : Recv message buffer count (only for two-sided transfers). Default is %u\n", L_recv_buffer_count);
  printf("   -e          : Event driven completion notification. Default is polling\n");
  printf("   -o          : Switch to one-sided (endpoint B not involved in transfers). Default is '%s'\n", L_two_sided ? "two-sided" : "one-sided");
  printf("   -v          : Validate the messages. Default is '%s'\n", L_validate ? "yes" : "no");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printUsageAndExit(argv[0]);
  }

  // Process the arguments
  L_provider = argv[1];
  for (int i=2; i<argc; i++) {
    if (strcmp(argv[i], "-e") == 0) {
      L_use_polling_completion = false;
    } else if (strcmp(argv[i], "-v") == 0) {
      L_validate = true;
    } else if (strcmp(argv[i], "-o") == 0) {
      L_two_sided = false;
    } else if (strncmp(argv[i], "-n=", 3) == 0) {
      int tokens = sscanf(argv[i], "-n=%u", &L_iterations);
      if (tokens != 1) { printf("Arg -n='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_iterations == 0) { printf("Arg -n='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-s=", 3) == 0) {
      int tokens = sscanf(argv[i], "-s=%u", &L_send_buffer_count);
      if (tokens != 1) { printf("Arg -s='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_send_buffer_count == 0) { printf("Arg -s='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-r=", 3) == 0) {
      int tokens = sscanf(argv[i], "-r=%u", &L_recv_buffer_count);
      if (tokens != 1) { printf("Arg -r='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_recv_buffer_count == 0) { printf("Arg -r='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-b=", 3) == 0) {
      int tokens = sscanf(argv[i], "-b=" UINT64_FORMAT, &L_message_bytes);
      if (tokens != 1) { printf("Arg -b='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else {
      printUsageAndExit(argv[0]);
    }
  }

  // Start threads
  pthread_t endpointA_thread_id;
  pthread_t endpointB_thread_id;
  pthread_create(&endpointA_thread_id, NULL, throughputThread, (void *)1LL);
  pthread_create(&endpointB_thread_id, NULL, throughputThread, NULL);

  // Wait for threads to complete processing
  pthread_join(endpointA_thread_id, NULL);
  pthread_join(endpointB_thread_id, NULL);

  return EXIT_SUCCESS;
}
