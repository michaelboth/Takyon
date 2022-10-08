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
#if defined(__APPLE__)
  #define UINT64_FORMAT "%llu"
#else
  #define UINT64_FORMAT "%ju"
#endif

static uint32_t L_iterations = 1000000;
static uint64_t L_message_bytes = 1024;
static uint32_t L_src_buffer_count = 10;
static uint32_t L_dest_buffer_count = 10;
static bool L_use_polling_completion = true;
static bool L_two_sided = true;
static bool L_validate = false;

static void printUsageAndExit(const char *program) {
  printf("usage: %s <A|B> \"<provider>\" [-h] [-i=<uint32>] [-bytes=<uint64>] [-sbufs=<uint32>] [-dbufs=<uint32>] [-e] [-write] [-V]\n", program);
  printf("   -h              : Print this message and exit\n");
  printf("   -i=<uint32>     : Number of messages to transfer. Default is %u\n", L_iterations);
  printf("   -bytes=<uint64> : Bytes per message. Can use 0 if two-sided and supported by provider. Default is " UINT64_FORMAT "\n", L_message_bytes);
  printf("   -sbufs=<uint32> : Source message buffer count. Default is %u\n", L_src_buffer_count);
  printf("   -dbufs=<uint32> : Destination message buffer count. Default is %u\n", L_dest_buffer_count);
  printf("   -e              : Event driven completion notification. Default is polling\n");
  printf("   -write          : Switch to one-sided (endpoint B not involved in transfers). Default is '%s'\n", L_two_sided ? "two-sided" : "one-sided");
  printf("   -V              : Validate the messages. Default is '%s'\n", L_validate ? "yes" : "no");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printUsageAndExit(argv[0]);
  }

  // Process the arguments
  if (strlen(argv[1]) != 1 || (argv[1][0] != 'A' && argv[1][0] != 'B')) { printf("First arg must be 'A' or 'B'.\n"); printUsageAndExit(argv[0]); }
  const bool is_endpointA = (argv[1][0] == 'A');
  const char *provider = argv[2];
  for (int i=3; i<argc; i++) {
    if (strcmp(argv[i], "-e") == 0) {
      L_use_polling_completion = false;
    } else if (strcmp(argv[i], "-V") == 0) {
      L_validate = true;
    } else if (strcmp(argv[i], "-write") == 0) {
      L_two_sided = false;
    } else if (strncmp(argv[i], "-i=", 3) == 0) {
      int tokens = sscanf(argv[i], "-i=%u", &L_iterations);
      if (tokens != 1) { printf("Arg -i='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_iterations == 0) { printf("Arg -i='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-sbufs=", 3) == 0) {
      int tokens = sscanf(argv[i], "-sbufs=%u", &L_src_buffer_count);
      if (tokens != 1) { printf("Arg -sbufs='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_src_buffer_count == 0) { printf("Arg -sbufs='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-dbufs=", 3) == 0) {
      int tokens = sscanf(argv[i], "-dbufs=%u", &L_dest_buffer_count);
      if (tokens != 1) { printf("Arg -dbufs='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
      if (L_dest_buffer_count == 0) { printf("Arg -dbufs='%s' must be greater than zero.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else if (strncmp(argv[i], "-bytes=", 3) == 0) {
      int tokens = sscanf(argv[i], "-bytes=" UINT64_FORMAT, &L_message_bytes);
      if (tokens != 1) { printf("Arg -bytes='%s' is invalid.\n", argv[i]); printUsageAndExit(argv[0]); }
    } else {
      printUsageAndExit(argv[0]);
    }
  }

  // Run one endpoint of the path
  throughput(is_endpointA, provider, L_iterations, L_message_bytes, L_src_buffer_count, L_dest_buffer_count, L_use_polling_completion, L_two_sided, L_validate);

  return EXIT_SUCCESS;
}
