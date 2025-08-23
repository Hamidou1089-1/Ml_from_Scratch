# Récupérer tous les fichiers sources
file(GLOB_RECURSE SOURCES "*.cxx" "*.cpp")

# Créer la bibliothèque principale
add_library(ml ${SOURCES})

# Inclure les headers
target_include_directories(ml PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/concepts
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/losses
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/losses/regularization
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/optimizers
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/models
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ml PUBLIC Eigen3::Eigen)

message(STATUS "Creating normal library with sources: ${SOURCES}")

# Ajouter les flags de profiling SI activés
if(ENABLE_PROFILING)
    target_compile_options(ml PRIVATE ${PROFILING_COMPILE_FLAGS})
    target_link_options(ml PRIVATE ${PROFILING_LINK_FLAGS})
    message(STATUS "ml: Profiling flags added")
endif()

# Propriétés de la bibliothèque
set_target_properties(ml PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)