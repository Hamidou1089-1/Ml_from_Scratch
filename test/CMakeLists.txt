file(GLOB_RECURSE TEST_SOURCES "*.cxx" "*.cpp")

foreach (text_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)

    add_executable(${test_name} ${test_file})

    target_link_libraries(${test_name} ml)

    target_include_directories(${test_name} PRIVATE ../include)

    if(ENABLE_PROFILING)
        target_compile_options(${test_name} PRIVATE ${PROFILING_COMPILE_FLAGS})
        target_link_options(${test_name} PRIVATE ${PROFILING_LINK_FLAGS})
        message(STATUS "${test_name}: Profiling flags added")
    endif ()
endforeach ()

