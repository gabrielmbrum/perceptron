#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <iostream>
#include <string>
#include <vector>

int sum_arrays(float *arr1, float *arr2, int size);
int epoch(float *inputs, float *weights, float b, float lr, int expected_output, int size);
float u(float *inputs, float *weights, float b, int size);
int g_u(float u);
void intro(float *inputs, float *weights, float b, float lr, int expected_output, int size);

#endif
