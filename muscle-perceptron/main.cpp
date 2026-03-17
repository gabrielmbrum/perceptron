#include "perceptron.hpp"
#include <iostream>

using namespace std;

int main() {
    int* answers;
    answers = (int*) malloc(3 * sizeof(int));

    intro();

    forms(&answers);

    cout << "\noutput: " << (calculate_output(answers) > 0 ? "you will build muscles!" : "you wont build muscles") << "\n";

    free(answers);
    return 0;
}
