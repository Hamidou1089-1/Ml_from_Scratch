# Récupérer tous les fichiers de test
file(GLOB_RECURSE TEST_SOURCES "*.cxx" "*.cpp")

# Créer un exécutable pour chaque fichier de test
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    
    add_executable(${test_name} ${test_file})
    
    # Lier avec la bibliothèque ml
    target_link_libraries(${test_name} PRIVATE ml)
    
    # Headers spécifiques aux tests si nécessaire
    target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    
    # Propriétés des tests
    set_target_properties(${test_name} PROPERTIES
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    message(STATUS "Created test executable: ${test_name}")
endforeach()

# Information selon le mode
if(ENABLE_PROFILING)
    message(STATUS "All test executables: Configured for PROFILING")
    message(STATUS "Usage: Run executable, then 'gprof ./executable gmon.out > profile.txt'")
else()
    message(STATUS "All test executables: Configured for PERFORMANCE")
    message(STATUS "Usage: Run 'time ./executable' for performance measurement")
endif()

# Optionnel : ajouter les tests au système de test de CMake
enable_testing()
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()