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

#include "utils_arg_parser.h"
#include "takyon.h" // For TAKYON_MAX_INTERCONNECT_CHARS
#include <string.h>
#include <stdio.h>

// This is a simple parse with a few requirements
//  - Format: "interconnect [-arg[=value]]*"
//  - Args are separated by a single space: spaces are not allowed in values
//  - Flags are always the form " -<named_flag>"
//  - Name/value pairs are always the format: " -<arg_name>=<value>"
//  - Full example: "socket -IP=192.168.0.21 -port=23454"

static bool getArgWithName(const char *arguments, const char *key, bool is_prefix, bool *found_ret, char *result, const int max_chars, char *error_message, int max_error_message_chars) {
  // Skip any leading spaces
  while (arguments[0] == ' ') arguments++;

  // Scan all arguments
  const char *space_ptr = strchr(arguments, ' '); // Skipping first argument
  while (space_ptr != NULL) {
    // Skip any leading spaces
    while (space_ptr[0] == ' ') space_ptr++;
    if (space_ptr[0] == '\0') break;
 
    // Found an arg, copy it to the temp pointer
    int index = 0;
    while (space_ptr[index] != ' ' && space_ptr[index] != '\0' && index < (max_chars-1)) {
      result[index] = space_ptr[index];
      index++;
    }
    result[index] = '\0';

    // See if it's the correct key=value
    if (is_prefix && strncmp(key, result, strlen(key))==0) {
      // Found the prefix.
      *found_ret = true;
      // Verify value exists for the key
      if (strlen(key) == strlen(result)) {
        if (error_message != NULL) snprintf(error_message, max_error_message_chars, "Argument '%s' missing its value in argument list '%s'", key, arguments);
        return false;
      }
      // Get the value
      int index2 = (int)strlen(key);
      index = 0;
      while (result[index2] != '\0') {
        result[index] = result[index2];
        index++;
        index2++;
      }
      result[index] = '\0';
      return true;
    }

    // See if it's the correct flag
    if (!is_prefix && strcmp(key, result)==0) {
      // Found the flag
      *found_ret = true;
      return true;
    }

    // Continue looking
    space_ptr++;
    space_ptr = strchr(space_ptr, ' ');
  }

  *found_ret = false;
  return true;
}

bool argGetInterconnect(const char *arguments, char *result, const int max_chars, char *error_message, int max_error_message_chars) {
  // Skip any leading spaces
  while (arguments[0] == ' ') arguments++;
  // Copy first argument
  int index = 0;
  while (arguments[index] != ' ' && arguments[index] != '\0' && index < (max_chars-1)) {
    result[index] = arguments[index];
    index++;
  }
  result[index] = '\0';
  if (strlen(result) == 0) {
    snprintf(error_message, max_error_message_chars, "No interconnect specified.");
    return false;
  }
  return true;
}

bool argGetFlag(const char *arguments, const char *name) {
  bool found = false;
  char value_text[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!getArgWithName(arguments, name, false, &found, value_text, TAKYON_MAX_INTERCONNECT_CHARS, NULL, 0)) return false;
  return found;
}

bool argGetText(const char *arguments, const char *name, char *result, const int max_chars, bool *found_ret, char *error_message, int max_error_message_chars) {
  if (!getArgWithName(arguments, name, true, found_ret, result, max_chars, error_message, max_error_message_chars)) return false;
  if (!(*found_ret)) return true; // Argument not found
  return true;
}

bool argGetInt(const char *arguments, const char *name, int *result, bool *found_ret, char *error_message, int max_error_message_chars) {
  char value_text[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!getArgWithName(arguments, name, true, found_ret, value_text, TAKYON_MAX_INTERCONNECT_CHARS, error_message, max_error_message_chars)) return false;
  if (!(*found_ret)) return true; // Argument not found
  int tokens = sscanf(value_text, "%d", result);
  if (tokens != 1) {
    snprintf(error_message, max_error_message_chars, "Value for arg '%s' is not an int: '%s'.", name, value_text);
    return false;
  }
  return true;
}

bool argGetUInt(const char *arguments, const char *name, uint32_t *result, bool *found_ret, char *error_message, int max_error_message_chars) {
  char value_text[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!getArgWithName(arguments, name, true, found_ret, value_text, TAKYON_MAX_INTERCONNECT_CHARS, error_message, max_error_message_chars)) return false;
  if (!(*found_ret)) return true; // Argument not found
  int tokens = sscanf(value_text, "%u", result);
  if (tokens != 1) {
    snprintf(error_message, max_error_message_chars, "Value for arg '%s' is not an unsigned int: '%s'.", name, value_text);
    return false;
  }
  return true;
}

bool argGetFloat(const char *arguments, const char *name, float *result, bool *found_ret, char *error_message, int max_error_message_chars) {
  char value_text[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!getArgWithName(arguments, name, true, found_ret, value_text, TAKYON_MAX_INTERCONNECT_CHARS, error_message, max_error_message_chars)) return false;
  if (!(*found_ret)) return true; // Argument not found
  int tokens = sscanf(value_text, "%f", result);
  if (tokens != 1) {
    snprintf(error_message, max_error_message_chars, "Value for arg '%s' is not a float: '%s'.", name, value_text);
    return false;
  }
  return true;
}
