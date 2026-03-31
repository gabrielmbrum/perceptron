#include <iostream>
#include "perceptron.hpp"

using namespace std;

string param_names[3] = {"working out", "eating healthy", "playing sports"};
int weights[3] = {6, 4, 3};
int threshold = sum_array(weights) / 2;

int sum_array(int *arr) {
  int total = 0;
  for (int i = 0; i < 3; i++) {
    total += arr[i];
  }
  return total;
}

int calculate_output(int *answers) {
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += weights[i] * answers[i];
  }
  
  return (sum >= threshold) ? 1 : 0;
}

void intro() {
  cout << "-------------------------------------\n";
  cout << "welcome to perceptron simulator!";

  cout << "\nparameters and its weights:\n";
  for (int i = 0; i < 3; i++) {
    cout << "-> param: " << param_names[i] << " | weight: " << weights[i] << "\n";
  }

  cout << "\nthreshold: " << threshold << "\n";
  cout << "-------------------------------------\n";
}

void forms(int **answers) {
  cout << "please answer the following questions with 0 (no) or 1 (yes):\n";
  
  for (int i = 0; i < 3; i++) {
    cout << "are you " << param_names[i] << "? ";
    cin >> (*answers)[i];
    
    // Validate input
    while ((*answers)[i] != 0 && (*answers)[i] != 1) {
      cout << "invalid input!!! please enter 0 (no) or 1 (yes): ";
      cin >> (*answers)[i];
    }
  }
}
