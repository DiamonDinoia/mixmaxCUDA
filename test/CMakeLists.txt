include(CheckLanguage)

option(GPU "Enable GPU" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES ON)


if ("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++23 standard")
    set(CMAKE_CXX_STANDARD 23)
elseif ("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++20 standard")
    set(CMAKE_CXX_STANDARD 20)
elseif ("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CMAKE_CXX_STANDARD 17)
    message(STATUS "Detected support for C++17 standard")
else ()
    message(FATAL_ERROR "Cannot detect CXX_STANDARD of C++17 or newer.")
endif ()

if (GPU)
    check_language(CUDA)
    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Requested CUDA support, but no CMAKE_CUDA_COMPILER available")
    endif ()
    enable_language(CUDA)
    if ("cuda_std_23" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
        set(CMAKE_CUDA_STANDARD 23)
        message(STATUS "Detected support for CUDA23 standard")
    elseif ("cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
        set(CMAKE_CUDA_STANDARD 20)
        message(STATUS "Detected support for CUDA20 standard")
    elseif ("cuda_std_17" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
        set(CMAKE_CUDA_STANDARD 17)
        message(STATUS "Detected support for CUDA17 standard")
    else ()
        message(FATAL_ERROR "Cannot detect CUDA_STANDARD of C++17 or newer.")
    endif ()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif ()

function(add_compilation_flags variableName)
    set(old_flags ${${variableName}})
    message(STATUS "COMPILATION FLAGS ${old_flags}")
    if (GPU)
        if (NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            string(REPLACE "-pedantic" "-Wno-pedantic" CUDA_flags "${CUDA_flags}")
            add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${old_flags}>$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_flags}>")
        endif ()
    else ()
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${old_flags}>")
    endif ()
endfunction()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(FLAGS -O0 -fstack-protector-all -Wall -Wextra -pedantic)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-G>")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(FLAGS -O3 -march=native)
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    add_definitions(-DDEBUG)
    set(FLAGS -O3 -g -fno-omit-frame-pointer -march=native)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-G>")
endif ()
add_compilation_flags(FLAGS)

if (GPU)
    if (${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
        cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
        message("ARCH_FLAGS = ${ARCH_FLAGS}")
        string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
        string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
    else ()
        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
        string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
        string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
        SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
        set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    endif ()
    list(APPEND GPU_FLAGS --expt-relaxed-constexpr --ptxas-options=-v)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${GPU_FLAGS}>")
    find_package(CUDAToolkit REQUIRED)
    message("CUDA INCLUDE ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION_MAJOR}")
    message(STATUS "CUDA Path: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA CUDAToolkit include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    list(APPEND LIBRARIES CUDA::cudart_static CUDA::curand_static)
    message("GPU LIBRARIES: ${LIBRARIES}")
endif ()

set(CPM_DOWNLOAD_VERSION 0.36.0)
if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME MixMaxOriginal
        URL http://www.hepforge.org/archive/mixmax/mixmax_release_200final.zip
        DOWNLOAD_ONLY YES
        EXCLUDE_FROM_ALL YES
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mixmax_include)
file(COPY ${MixMaxOriginal_SOURCE_DIR}/mixmax.hpp
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mixmax_include)
file(COPY ${MixMaxOriginal_SOURCE_DIR}/mixmax_skip_N240.c
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mixmax_include)
file(COPY ${MixMaxOriginal_SOURCE_DIR}/mixmax_skip_N17.c
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mixmax_include)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/mixmax_include)

add_executable(main main.cpp)
target_link_libraries(main mixmax)

if (GPU)
    add_executable(main_gpu main.cu)
    target_link_libraries(main_gpu mixmax)
endif ()