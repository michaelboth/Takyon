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
#include <pthread.h>

static const char *L_provider = NULL;
static uint32_t L_iterations = 1;

static void *throughputThread(void *user_data) {
  bool is_endpointA = (user_data != NULL);
  throughput(is_endpointA, L_provider, L_iterations);
  return NULL;
}

int main(int argc, char **argv) {
  if (argc != 3) { printf("usage: %s \"<provider>\" <iterations>\n", argv[0]); return 1; }
  L_provider = argv[1];
  L_iterations = (uint32_t)atoi(argv[2]);

  // Start threads
  pthread_t endpointA_thread_id;
  pthread_t endpointB_thread_id;
  pthread_create(&endpointA_thread_id, NULL, throughputThread, (void *)1LL);
  pthread_create(&endpointB_thread_id, NULL, throughputThread, NULL);

  // Wait for threads to complete processing
  pthread_join(endpointA_thread_id, NULL);
  pthread_join(endpointB_thread_id, NULL);
  return 0;
}
