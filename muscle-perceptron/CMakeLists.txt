cmake_minimum_required(VERSION 3.10)

# Define the project name
project(RedesNeurais)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable for the perceptron example
add_executable(muscle-perceptron main.cpp perceptron.cpp)
