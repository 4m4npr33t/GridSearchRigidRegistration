cmake_minimum_required(VERSION 3.24)
project(rigid_registration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(cpu_registration cpu_main.cpp)
if(MSVC)
    target_compile_options(cpu_registration PRIVATE /W4)
else()
    target_compile_options(cpu_registration PRIVATE -Wall -Wextra)
endif()

# CUDA is optional: the CPU lesson can build on a machine without the toolkit.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # RTX 3060. Override with -DCMAKE_CUDA_ARCHITECTURES=... for another GPU.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    add_executable(cuda_registration main.cu)
    target_compile_options(cuda_registration PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>)
else()
    message(STATUS "CUDA compiler not found; building cpu_registration only")
endif()
