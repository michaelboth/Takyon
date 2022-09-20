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

#ifndef _takyon_inter_thread_manager_h_
#define _takyon_inter_thread_manager_h_

#include <stdbool.h>
#include "takyon.h"
#include <stdint.h>
#include <pthread.h>

typedef struct { // This is used to allow threads in the same process to share a mutex and conditional variable for synchronizing communications
  uint32_t provider_id; // Make sure this is different for each provider that uses this manager.
  uint32_t path_id;
  TakyonPath *pathA;
  TakyonPath *pathB;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool connected;
  bool disconnected;
  bool connection_broken;
  uint32_t usage_count;
} InterThreadManagerItem;

#ifdef __cplusplus
extern "C"
{
#endif

extern bool interThreadManagerInit();
extern void interThreadManagerFinalize();
extern InterThreadManagerItem *interThreadManagerConnect(uint32_t provider_id, uint32_t path_id, TakyonPath *path, double timeout_in_seconds);
extern void interThreadManagerMarkConnectionAsBad(InterThreadManagerItem *item);
extern bool interThreadManagerDisconnect(TakyonPath *path, InterThreadManagerItem *item, double timeout_in_seconds);

#ifdef __cplusplus
}
#endif

#endif
