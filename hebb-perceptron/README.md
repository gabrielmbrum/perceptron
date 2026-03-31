# hebb-perceptron

## about

this perceptron implements the **hebb rule** for adjusting the weigths of the features.

beyond that, a new parameter is added to the $g(u)$ function: `the bias`.

it is a measure that says how easy it is to get the perceptron to output a 1

<p align="center">
  <img src="assets/image1.png" alt="output formula">
</p>

a example of a percpetron with two inputs with $weights = 2$ and $bias = 3$, we have:

<p align="center">
  <img src="assets/image2.png" alt="percep diagram">
</p>

(this diagram shows a NAND logic gate)

## concepts

### hebb-rule

the hebb rule adjust the weights of a perceptron based on the **learning rate** and the **error**.

`learning rate` is a hiperparameter, which says how much the weights will be changed (or you can think as how much the percpetron will learn)

`error` is the difference of the expected output with the percpeptron output, so how we are using binary outputs, the error can be: -1, 0 or 1.

this rule can be applied to the bias too, but only in the classification perceptron we use it.

the formuila is:

$$weight_{i + 1} = weights_{i} + lr * (expected_output - output) * input_{i}$$

### $u$ & $g(u)$ 
here the perceptron has a new attribute: the bias.

the bias is important because it improves the capacity of learning of the perceptron. it's used for the output calculation:

$$u = (\sum_{i=1}^{n} weights[i] \cdot inputs[i]) - bias $$

and then we calculate the output

$$
g(u) = 
\begin{cases} 
        1, & \text{if } u \ge 0 \\ 
        0, & \text{if } u < 0 
\end{cases}
$$

## how to run

```bash
cmake -S . -B build
cmake --build build
./build/hebb-perceptron 
```