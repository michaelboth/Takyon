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

#include "utils_time.h"
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <time.h>

void clockSleep(int64_t microseconds) {
  if (microseconds >= 0) {
    usleep(microseconds);
  }
}

void clockSleepYield(int64_t microseconds) {
  /*
   * This is used to facilitate waiting with a context switch.
   * This will give other lower priority tasks the ability to
   * get some CPU time.
   */
  if (microseconds > 0) {
    usleep(microseconds);
  } else {
    /* Need to at least do a context switch out to give some other process a try (not a true thread yield) */
    sched_yield();
  }
}

int64_t clockTimeNanoseconds() {
  /*+ base time */
  int64_t total_nanoseconds;
  struct timeval tp;
  /*+ clock_gettime? */
  gettimeofday(&tp, NULL);
  total_nanoseconds = ((int64_t)tp.tv_sec * 1000000000LL) + (int64_t)(tp.tv_usec * 1000);
  return total_nanoseconds;
}

double clockTimeSeconds() {
  static long long base_time;
  static int got_base_time = 0;
  struct timespec curr_time;
  clock_gettime(CLOCK_MONOTONIC, &curr_time); // CLOCK_MONOTONIC is only increasing
  long long total_nanoseconds = ((long long)curr_time.tv_sec * 1000000000LL) + (long long)curr_time.tv_nsec;
  if (!got_base_time) {
    base_time = total_nanoseconds;
    got_base_time = 1;
  }
  return (total_nanoseconds - base_time)/1000000000.0;
}
