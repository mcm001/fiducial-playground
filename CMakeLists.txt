cmake_minimum_required(VERSION 3.20)

# Generate compile_commands.json for clangd
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pvcalib)

add_subdirectory(app)