cmake_minimum_required(VERSION 3.10) 

project(mlp LANGUAGES C)

# Compiler
set(CMAKE_C_COMPILER "icx")

# Compilation flags
add_compile_options(-march=native -Wall -Wextra -g3 -O3 -fiopenmp)

# HEADER	
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/mnist_reader
    ${PROJECT_SOURCE_DIR}/include/network
    ${PROJECT_SOURCE_DIR}/include/benchmark
)

# Définir où trouver les fichiers sources
file(GLOB_RECURSE SOURCES "src/*.c")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})  # MLP 

add_executable(mlp ${SOURCES})

# Link
target_link_libraries(mlp m)
target_link_options(mlp PRIVATE -qnextgen -qmkl)
