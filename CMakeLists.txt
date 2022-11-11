
cmake_minimum_required(VERSION 3.19)
project(mixmax VERSION 1.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(TESTS "Enable TESTS" OFF)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif ()


add_library(mixmax INTERFACE)
target_include_directories(mixmax INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(mixmax INTERFACE cxx_std_11) # This means at least C++11

install(TARGETS mixmax)

if (TESTS)
    add_subdirectory(test)
endif ()