# Récupérer tous les fichiers sources
file(GLOB_RECURSE SOURCES "*.cxx" "*.cpp")

# Créer la bibliothèque principale
add_library(ml ${SOURCES})

# Inclure les headers
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/concepts
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/losses
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/losses/regularization
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/optimizers
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/models
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(ml PUBLIC ${INCLUDE_DIRS})
target_link_libraries(ml PUBLIC Eigen3::Eigen)

# Propriétés de la bibliothèque
set_target_properties(ml PROPERTIES
    CXX_STANDARD 26
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Configuration selon le mode
if(ENABLE_PROFILING)
    message(STATUS "ml: Configured for PROFILING (no optimizations)")
else()
    message(STATUS "ml: Configured for PERFORMANCE (full optimizations)")
    
    # Optimisations spécifiques aux templates (seulement en mode performance)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(ml PRIVATE
            $<$<CXX_COMPILER_ID:GNU>:-ftemplate-backtrace-limit=0>
            $<$<CXX_COMPILER_ID:Clang>:-ftemplate-backtrace-limit=0>
        )
    endif()
    
    # Précompiled headers pour accélérer la compilation (seulement en mode performance)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
        target_precompile_headers(ml PRIVATE
            <Eigen/Dense>
            <vector>
            <memory>
            <concepts>
            <type_traits>
        )
    endif()
endif()

message(STATUS "Sources found: ${SOURCES}")