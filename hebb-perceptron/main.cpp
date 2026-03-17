#include <iostream>
#include "perceptron.hpp"

using namespace std;

int main() {
    float* inputs;
    float* weights;
    float b;
    float lr;
    int d = 1; //expected value

    inputs = (float*) malloc(2 * sizeof(float));
    weights = (float*) malloc(2 * sizeof(float));

    inputs[0] = 0.2;
    inputs[1] = 0.8;
    weights[0] = 0.5;
    weights[1] = 0.1;

    b = 0.3;
    lr = 0.1;

    intro(inputs, weights, b, lr, d, 2);

    for (int i = 0; epoch(inputs, weights, b, lr, d, 2) == 0; i++) {
        cout << "Epoch " << i << endl;
    }

    cout << "Updated weights: " << weights[0] << ", " << weights[1] << endl;
    cout << "Final 'u' value: " << u(inputs, weights, b, 2) << endl;
    cout << "Final output: " << g_u(u(inputs, weights, b, 2)) << endl;

}