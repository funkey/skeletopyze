define_module(skeletopyze LIBRARY LINKS imageprocessing boost-python)
add_custom_target(rename_skeletopyze_lib ALL COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/python/libskeletopyze.so ${CMAKE_BINARY_DIR}/python/skeletopyze.so)
add_dependencies(rename_skeletopyze_lib skeletopyze)
