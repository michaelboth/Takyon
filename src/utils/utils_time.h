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

#ifndef _utils_time_h_
#define _utils_time_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern void clockSleepSeconds(double seconds);
extern void clockSleepUsecs(int64_t microseconds);
extern void clockSleepYield(int64_t microseconds);  // Goal is to force a context switch to give other threads time to process
extern int64_t clockTimeNanoseconds();              // Since some base time
extern double clockTimeSeconds();                   // Since some base time

#ifdef __cplusplus
}
#endif

#endif
