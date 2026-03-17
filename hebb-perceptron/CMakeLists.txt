cmake_minimum_required(VERSION 3.10)
project(hebb-perceptron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(hebb-perceptron main.cpp perceptron.cpp perceptron.hpp)
