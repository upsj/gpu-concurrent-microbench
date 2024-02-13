cmake_minimum_required(VERSION 3.23)
project(ConcurrentMicrobench LANGUAGES CXX CUDA)

option(USE_HIP "Use hip" OFF)
set(CMAKE_CXX_STANDARD 17)

if(USE_HIP)
    set(GPU_LANG HIP)
else()
    set(GPU_LANG CUDA)
endif()
enable_language(${GPU_LANG})

add_subdirectory(gpubench)

add_executable(benchmark benchmark.gpu.cpp)
set_source_files_properties(benchmark.gpu.cpp PROPERTIES LANGUAGE ${GPU_LANG})
target_link_libraries(benchmark nvbench::main)
