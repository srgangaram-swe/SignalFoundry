cmake_minimum_required(VERSION 3.20)
project(alphaforge_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Native benchmark binary (no Python required).
add_executable(bench_orderbook benchmarks/bench_main.cpp)
target_compile_options(bench_orderbook PRIVATE -O3)

# Python extension module (requires `pip install pybind11`).
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  pybind11_add_module(alphaforge_native src/bindings.cpp)
  target_compile_options(alphaforge_native PRIVATE -O3)
  # Drop the built module inside the Python package.
  set_target_properties(alphaforge_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../alphaforge)
else()
  message(STATUS "pybind11 not found — building benchmark only. "
                 "Use scripts/build_native.py for the Python extension.")
endif()
