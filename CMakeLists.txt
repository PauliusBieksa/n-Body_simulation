cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nbody)

#### Setup directories ####
#Main output directory
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/bin")
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/lib")

#### Add external dependencies ####
add_subdirectory("Lib/graphics")
set(GRAPHICS "Lib/graphics")
set(GIF_ENCODER "lib/gif")

#Glew is dumb and won't listen to our build directories, so manually set it.
set_target_properties(libglew_shared PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
)

#### Sequential implementation ####
file(GLOB_RECURSE SOURCES sequential/*.cpp sequential/*.h lib/gif/*.h)
add_executable(SEQUENTIAL ${SOURCES})
target_include_directories(SEQUENTIAL PRIVATE ${GRAPHICS} ${GIF_ENCODER})
target_link_libraries(SEQUENTIAL enu_graphics_framework)

#### Target for copying resources to build dir####
add_custom_target(copy_res ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/res" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/res")
