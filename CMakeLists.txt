cmake_minimum_required(VERSION 3.20)
project(ML)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OPTION pour le profiling
option(ENABLE_PROFILING "Enable gprof profiling" OFF)

# Si profiling activé, définir les flags
if(ENABLE_PROFILING)
    message(STATUS "Profiling enabled - adding -pg flags")
    set(PROFILING_COMPILE_FLAGS "-pg")
    set(PROFILING_LINK_FLAGS "-pg")
else()
    message(STATUS "Profiling disabled")
    set(PROFILING_COMPILE_FLAGS "")
    set(PROFILING_LINK_FLAGS "")
endif()

# Options de compilation
set(CMAKE_CXX_FLAGS_DEBUG "-g -o0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-o3 -DNDEBUG")



ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(demo)
ADD_SUBDIRECTORY(doc)
