# muscle-perceptron

>*are you a fake natty?*

here i coded a simple perceptron with 3 params, and its respective weights.

```c++
string param_names[3] = {"working out", "eating healthy", "playing sports"};
int weights[3] = {6, 4, 3};
int threshold = sum_array(weights) / 2;
```

the diagram of the perceptron is like:

<p align="center">
  <img src="assets/image2.png" alt="perceptron diagram">
</p>

the `threshold` is the limiar of the perceptron, that means that what decides if the output of the perceptron will be 0 or 1 is if the $w * x$ is greater than, or equal, to the threshold

<p align="center">
  <img src="assets/image1.png" alt="output calculation">
</p>