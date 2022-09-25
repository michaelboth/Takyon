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
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

static uint32_t L_iterations = 1000000;
static uint64_t L_message_bytes = 1024;
static uint32_t L_send_buffer_count = 10;
static uint32_t L_recv_buffer_count = 10;
static bool L_use_polling_completion = true;
static bool L_two_sided = true;
static bool L_validate = false;

static void printUsageAndExit(const char *program) {
  printf("usage: %s <A|B> \"<provider>\" [-h] [-n=<uint32>] [-b=<uint64>] [-s=<uint32>] [-r=<uint32>] [-e] [-o] [-v]\n", program);
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
  if (argc < 3) {
    printUsageAndExit(argv[0]);
  }

  // Process the arguments
  assert(strlen(argv[1]) == 1 && (argv[1][0] == 'A' || argv[1][0] == 'B'));
  const bool is_endpointA = (argv[1][0] == 'A');
  const char *provider = argv[2];
  for (int i=3; i<argc; i++) {
    if (strcmp(argv[i], "-e") == 0) {
      L_use_polling_completion = false;
    } else if (strcmp(argv[i], "-v") == 0) {
      L_validate = true;
    } else if (strcmp(argv[i], "-o") == 0) {
      L_two_sided = false;
    } else if (strncmp(argv[i], "-n=", 3) == 0) {
      int tokens = sscanf(argv[i], "-n=%u", &L_iterations);
      assert(tokens == 1);
      assert(L_iterations > 0);
    } else if (strncmp(argv[i], "-s=", 3) == 0) {
      int tokens = sscanf(argv[i], "-s=%u", &L_send_buffer_count);
      assert(tokens == 1);
      assert(L_send_buffer_count > 0);
    } else if (strncmp(argv[i], "-r=", 3) == 0) {
      int tokens = sscanf(argv[i], "-r=%u", &L_recv_buffer_count);
      assert(tokens == 1);
      assert(L_recv_buffer_count > 0);
    } else if (strncmp(argv[i], "-b=", 3) == 0) {
      int tokens = sscanf(argv[i], "-b=" UINT64_FORMAT, &L_message_bytes);
      assert(tokens == 1);
    } else {
      printUsageAndExit(argv[0]);
    }
  }

  // Run one endpoint of the path
  throughput(is_endpointA, provider, L_iterations, L_message_bytes, L_send_buffer_count, L_recv_buffer_count, L_use_polling_completion, L_two_sided, L_validate);

  return EXIT_SUCCESS;
}
