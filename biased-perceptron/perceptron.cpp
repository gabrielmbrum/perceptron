#include "perceptron.hpp"
#include <iostream>

int multiply_arrays(int *arr1, int *arr2, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr1[i] * arr2[i];
    }
    return sum;
}