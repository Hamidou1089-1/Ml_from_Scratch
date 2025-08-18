# Chercher Doxygen
find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
    # Configuration Doxygen
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)

    # Si tu as un fichier doxyfile simple (sans .in)
    set(DOXYGEN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile)

    # Créer la target pour générer la doc
    add_custom_target(doc
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Generating documentation with Doxygen"
            VERBATIM
    )

    # Optionnel : générer la doc automatiquement
    # add_custom_target(doc ALL
    #     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_CONFIG}
    #     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    #     COMMENT "Generating documentation with Doxygen"
    # )
else()
    message(WARNING "Doxygen not found, documentation will not be built")
endif()