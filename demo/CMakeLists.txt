# Récupérer tous les démos
file(GLOB_RECURSE DEMO_SOURCES "*.cxx" "*.cpp")

# Créer un exécutable pour chaque démo
foreach(demo_file ${DEMO_SOURCES})
    # Extraire le nom du fichier sans extension
    get_filename_component(demo_name ${demo_file} NAME_WE)

    # Créer l'exécutable
    add_executable(${demo_name} ${demo_file})

    # Lier avec la bibliothèque principale
    target_link_libraries(${demo_name} ml)

    # Inclure les headers
    target_include_directories(${demo_name} PRIVATE ../include)
endforeach()