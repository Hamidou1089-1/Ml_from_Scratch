# Récupérer tous les fichiers sources
file(GLOB_RECURSE SOURCES "*.cxx" "*.cpp")

# Créer la bibliotheque principale
add_library(ml ${SOURCES})

# Inclure les headers
target_include_directories(ml PUBLIC
        ../include
        ${CMAKE_CURRENT_SOURCE_DIR}
)


# Ajouter les flags de profiling SI activés
if(ENABLE_PROFILING)
    target_compile_options(ml PRIVATE ${PROFILING_COMPILE_FLAGS})
    target_link_options(ml PRIVATE ${PROFILING_LINK_FLAGS})
    message(STATUS "ml: Profiling flags added")
endif()

# Propriétés de la bibliotheque
set_target_properties(ml PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)