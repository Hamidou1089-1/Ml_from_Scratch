cmake_minimum_required(VERSION 3.20)
project(ML LANGUAGES CXX)

# Standards C++
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Trouver Eigen
find_package(Eigen3 REQUIRED)

# Options
option(ENABLE_PROFILING "Enable gprof profiling (disables optimizations)" OFF)
option(ENABLE_NATIVE_ARCH "Enable native architecture optimizations" ON)
option(ENABLE_LTO "Enable Link Time Optimization" OFF)

# Configuration CONDITIONNELLE selon profiling ou performance
if(ENABLE_PROFILING)
    message(STATUS "=== PROFILING MODE ===")
    message(WARNING "Profiling enabled - All optimizations disabled for accurate profiling")
    
    # Mode profiling : AUCUNE optimisation
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pg")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 -DNDEBUG -pg")  # Même en Release !
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    
    # Forcer le build type à Debug pour plus de clarté
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    
else()
    message(STATUS "=== PERFORMANCE MODE ===")
    message(STATUS "Profiling disabled - Full optimizations enabled")
    
    # Mode performance : optimisations maximales
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wconversion -Wshadow")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
    
    # Optimisations spécifiques aux templates (seulement en mode performance)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(TEMPLATE_OPTIMIZATION_FLAGS 
            "-finline-functions"
            "-ftemplate-backtrace-limit=0"
            "-fconstexpr-depth=1024"
            "-ftemplate-depth=1024"
        )
        
        # Architecture native pour vectorisation
        if(ENABLE_NATIVE_ARCH)
            list(APPEND TEMPLATE_OPTIMIZATION_FLAGS "-march=native" "-mtune=native")
            message(STATUS "Native architecture optimizations enabled")
        endif()
        
        # Ajouter aux flags de release
        string(JOIN " " TEMPLATE_FLAGS_STR ${TEMPLATE_OPTIMIZATION_FLAGS})
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${TEMPLATE_FLAGS_STR}")
        
        # Link Time Optimization (seulement en mode performance)
        if(ENABLE_LTO)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
            message(STATUS "Link Time Optimization enabled")
        endif()
    endif()
endif()

# Information de build
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling: ${ENABLE_PROFILING}")
if(NOT ENABLE_PROFILING)
    message(STATUS "Native arch: ${ENABLE_NATIVE_ARCH}")
    message(STATUS "LTO: ${ENABLE_LTO}")
endif()
message(STATUS "CXX Flags Debug: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CXX Flags Release: ${CMAKE_CXX_FLAGS_RELEASE}")

# Sous-répertoires
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(demo)
add_subdirectory(doc)