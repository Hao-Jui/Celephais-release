cmake_minimum_required(VERSION 3.25)
project(CelephaisSolutionConsumer LANGUAGES CXX)

find_package(Celephais CONFIG REQUIRED)

add_executable(solution_reader reader.cpp)
target_link_libraries(solution_reader PRIVATE Celephais::solution)
