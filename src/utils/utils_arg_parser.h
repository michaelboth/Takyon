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

#ifndef _utils_arg_parser_h_
#define _utils_arg_parser_h_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern bool argGetInterconnect(const char *arguments, char *result, const int max_chars, char *error_message, int max_error_message_chars);
extern bool argGetFlag(const char *arguments, const char *name);
extern bool argGetText(const char *arguments, const char *name, char *result, const int max_chars, bool *found_ret, char *error_message, int max_error_message_chars);
extern bool argGetInt(const char *arguments, const char *name, int *result, bool *found_ret, char *error_message, int max_error_message_chars);
extern bool argGetUInt(const char *arguments, const char *name, uint32_t *result, bool *found_ret, char *error_message, int max_error_message_chars);
extern bool argGetFloat(const char *arguments, const char *name, float *result, bool *found_ret, char *error_message, int max_error_message_chars);

#ifdef __cplusplus
}
#endif

#endif
