cmake_minimum_required(VERSION 3.20)
project(parallel_programming)

set(CMAKE_CXX_STANDARD 14)

add_executable(parallel_programming main.cpp parallel_sort.cpp)
