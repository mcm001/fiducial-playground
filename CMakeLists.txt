cmake_minimum_required(VERSION 3.20)

# Generate compile_commands.json for clangd
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

project(pvcalib)

add_subdirectory(app)