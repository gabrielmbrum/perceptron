# perceptron

perceptron is an abstraction of a neuron, its the basic cell of all the giants neural networks that can do amazing things.

if you want to learn about this basic cell, this repo is for you! (as it is for me too :>

here, i will be coding some code examples of a perceptron (or multiple perceptrons) that are very understandable.

# muscle-perceptron

this is the most basic perceptron example, is the easiest to understand, and this perceptron will tell if you will build muscles or not.

here we have three parameters {"working out", "eating healthy", "playing sports"}, the inputs are binary values (yes or no), then i associated weights for each parameter, the bigger the weight, more influent it will be to the output decision.

the output is decided if the sum of the multiplication of the vector of inputs by their weights is bigger than the threshold.

# hebb-perceptron

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

here, at the case of calculating the wrong desired output the bias is not adjusted, just the weights are update, followwing the **Hebb-Rule**:

$$weight_{i + 1} = weights_{i} + lr * (expected_output - output) * input_{i}$$

# classification-percpeptron

here we used three datasets to train and test in different scenarios.

the dataset 1 and 2 are 2D problems, but the dataset 3 has a 10 dimensions problem.

the weights are adjusted as in hebb-perceptron, but here the bias is also adjusted as:

$$
bias_{i+1} = bias_{i} + lr * error * (-1)
$$

the error is define by the difference between the desired output and the perceptron output.

# references

Neural Networks and Deep Learning - Michael Nielsen [http://neuralnetworksanddeeplearning.com/]

Master's Degree Classes - Prof. Dr. Lucas C. Ribas [https://www.ibilce.unesp.br/#!/departamentos/cienc-comp-estatistica/docentes/lucas-correia-ribas/]
