#include "perceptron.hpp"
#include <iostream>

using namespace std;

int multiply_arrays(float *arr1, float *arr2, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr1[i] * arr2[i];
    }
    return sum;
}

float u(float *inputs, float *weights, float b, int size) {
    float result = multiply_arrays(inputs, weights, size) - b;

    return result;   
}

int g_u(float u) {
    return u > 0 ? 1 : 0;
}

int epoch(float *inputs, float *weights, float b, float lr, int expected_output, int size) {
    int output = g_u(u(inputs, weights, b, size)) > 0 ? 1 : 0;

    if (output != expected_output) {
        for (int i = 0; i < size; i++) {
            weights[i] += lr * (expected_output - output) * inputs[i];
        }
        return 0;
    }

    return 1;
}

void intro(float *inputs, float *weights, float b, float lr, int expected_output, int size) {
    cout << "----------------------------\n";
    cout << "Biased Perceptron Algorithm\n";
    cout << "Inputs: ";
    for (int i = 0; i < size; i++) {
        cout << inputs[i] << " ";
    }
    cout << "\nWeights: ";
    for (int i = 0; i < size; i++) {
        cout << weights[i] << " ";
    }
    cout << "\nBias: " << b << "\nLearning Rate: " << lr << "\nExpected Output: " << expected_output << std::endl;
    cout << "----------------------------\n\n";
}