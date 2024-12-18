# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Find GraphScope module installed by Pypi
# This module defines
#  GRAPHSCOPE_FOUND, whether GraphScope has been found
#  GRAPHSCOPE_HOME, path for GraphScope module
#  GRAPHSCOPE_ANALYTICAL_INCLUDE_DIR, directory containing analytical headers
#  GRAPHSCOPE_ANALYTICAL_LIBS, path to analytical shared libraries (gs_proto gs_util)
#  GRAPHSCOPE_VERSION, version of found GraphScope, not support yet!

set(GRAPHSCOPE_SEARCH_LIB_PATH_SUFFIXES)
if (CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND GRAPHSCOPE_SEARCH_LIB_PATH_SUFFIXES "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
endif ()
list(APPEND GRAPHSCOPE_SEARCH_LIB_PATH_SUFFIXES
            "lib64"
            "lib32"
            "lib")

# Add this block before calling find_package(Python)
if (DEFINED ENV{CONDA_PREFIX})
  set(Python_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()

# Continue with the original find_package call, now with the correct executable
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set shared library name for ${base_name} to ${output_variable}
#
# Example:
#   graphscope_shared_library_name(GRAPHSCOPE_UTIL_SHARED_LIBRARY_NAME gs_util)
#   # -> GRAPHSCOPE_UTIL_SHARED_LIBRARY_NAME = libgs_util.so on Linux
#   # -> GRAPHSCOPE_UTIL_SHARED_LIBRARY_NAME = libgs_util.dylib on macOS
function(graphscope_shared_library_name output_variable base_name)
  set(${output_variable}
    "${CMAKE_SHARED_LIBRARY_PREFIX}${base_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
    PARENT_SCOPE
  )
endfunction()

function(graphscope_find_package
         prefix
         home
         header_path)
  # include dir
  find_path(${prefix}_analytical_include_dir "${header_path}"
            PATHS "${home}"
            PATH_SUFFIXES "include"
            NO_DEFAULT_PATH)
  set(analytical_include_dir "${${prefix}_analytical_include_dir}")
  set(${prefix}_ANALYTICAL_INCLUDE_DIR "${analytical_include_dir}" PARENT_SCOPE)

  # libs
  graphscope_shared_library_name(util_lib_name gs_util)
  graphscope_shared_library_name(proto_lib_name gs_proto)
  set(${prefix}_analytical_libs)
  # gs_util
  find_library(${prefix}_analytical_util_lib
               NAMES "${util_lib_name}"
               PATHS "${home}"
               PATH_SUFFIXES ${GRAPHSCOPE_SEARCH_LIB_PATH_SUFFIXES}
               NO_DEFAULT_PATH)
  list(APPEND ${prefix}_analytical_libs ${${prefix}_analytical_util_lib})
  # gs_proto
  find_library(${prefix}_analytical_proto_lib
              NAMES "${proto_lib_name}"
              PATHS "${home}"
              PATH_SUFFIXES ${GRAPHSCOPE_SEARCH_LIB_PATH_SUFFIXES}
              NO_DEFAULT_PATH)
  list(APPEND ${prefix}_analytical_libs ${${prefix}_analytical_proto_lib})
  set(${prefix}_ANALYTICAL_LIBS "${${prefix}_analytical_libs}" PARENT_SCOPE)
endfunction()

# set GRAPHSCOPE_HOME from env
if (NOT "$ENV{GRAPHSCOPE_HOME}" STREQUAL "")
  file(TO_CMAKE_PATH "$ENV{GRAPHSCOPE_HOME}" GRAPHSCOPE_HOME)
endif ()

# or set GRPAHSCOPE_HOME with: site-packages/graphscope.runtime
if ("${GRAPHSCOPE_HOME}" STREQUAL "")
  # find graphscope from per user site-packages directory (PEP 370)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m site --user-site
    OUTPUT_VARIABLE PYTHON_SITE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(GRAPHSCOPE_HOME "${PYTHON_SITE_DIR}/graphscope.runtime")

  if (NOT EXISTS "${GRAPHSCOPE_HOME}")
    # find graphscope from global site-packages
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "from distutils import sysconfig as sc;print(sc.get_python_lib())"
      OUTPUT_VARIABLE PYTHON_SITE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(GRAPHSCOPE_HOME "${PYTHON_SITE_DIR}/graphscope.runtime")
  endif ()
endif ()

graphscope_find_package(GRAPHSCOPE "${GRAPHSCOPE_HOME}" graphscope/core/config.h)

string(FIND "${GRAPHSCOPE_ANALYTICAL_INCLUDE_DIR}" "NOTFOUND" GRAPHSCOPE_ANALYTICAL_INCLUDE_DIR_FOUND)
string(FIND "${GRAPHSCOPE_ANALYTICAL_LIBS}" "NOTFOUND" GRAPHSCOPE_ANALYTICAL_LIBS_NOTFOUND)

if (${GRAPHSCOPE_ANALYTICAL_INCLUDE_DIR_FOUND} STREQUAL "-1" AND ${GRAPHSCOPE_ANALYTICAL_LIBS_NOTFOUND} STREQUAL "-1")
  set(GRAPHSCOPE_FOUND ON)
  message(STATUS "Found the GraphScope HOME: ${GRAPHSCOPE_HOME}") 
  message(STATUS "Found the GraphScope analytical include dir: ${GRAPHSCOPE_ANALYTICAL_INCLUDE_DIR}")
  message(STATUS "Found the GraphScope analytical libs: ${GRAPHSCOPE_ANALYTICAL_LIBS}")
endif ()
