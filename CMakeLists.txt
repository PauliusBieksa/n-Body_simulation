cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(nbody LANGUAGES CXX CUDA)

find_package(CUDA)

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

#### OMP implementation ####
file(GLOB_RECURSE SOURCES1 OMP_parallel/*.cpp OMP_parallel/*.h lib/gif/*.h)
add_executable(OMP_PARALLEL ${SOURCES1})
target_include_directories(OMP_PARALLEL PRIVATE ${GRAPHICS} ${GIF_ENCODER})
target_link_libraries(OMP_PARALLEL enu_graphics_framework)

#
#add_library(cuda_stuff STATIC)
CUDA_ADD_LIBRARY(cuda_stuff STATIC)
target_sources(cuda_stuff PRIVATE ${CMAKE_SOURCE_DIR}/CUDA_parallel/CUDA_parallel.cu PUBLIC ${CMAKE_SOURCE_DIR}/CUDA_parallel/CUDA_parallel.cuh)
#set_target_properties(cuda_stuff PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/lib)
#set_target_properties(cuda_stuff PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

#### CUDA_PARALLEL implementation ####
file(GLOB_RECURSE SOURCES2 CUDA_parallel/*.cpp CUDA_parallel/*.h lib/gif/*.h)
add_executable(CUDA_PARALLEL ${SOURCES2})
target_include_directories(CUDA_PARALLEL PRIVATE ${GRAPHICS} ${GIF_ENCODER})
target_link_libraries(CUDA_PARALLEL enu_graphics_framework cuda_stuff)

# Makes sure cuda is compiles separately
#set_target_properties(CUDA_PARALLEL PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET CUDA_PARALLEL PROPERTY CUDA_SEPERABLE_COMPILATION ON)

#### Target for copying resources to build dir####
add_custom_target(copy_res ALL COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/res" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/res")
