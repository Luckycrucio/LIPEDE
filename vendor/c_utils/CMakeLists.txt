cmake_minimum_required(VERSION 3.16)
project(c_utils)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(c_gen_normal_map src/c_gen_normal_map.cpp)
target_link_libraries(c_gen_normal_map PRIVATE OpenMP::OpenMP_CXX)