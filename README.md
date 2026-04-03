# perceptron and adaline: artificial neural networks fundamentals

perceptron is an abstraction of a neuron, its the basic cell of all these giants neural networks that can do amazing things, and can be used to solve linear problems.

adaline is the evolution of the perceptron, which uses some different concepts that will be specified into this repo.

if you want to learn about this basic cells, this repo is for you! ;)

## about

in this repo we have 3 implementations of perceptrons and 1 of an adaline:

- the [muscle-perceptron](./muscle-perceptron/) is a very basic example of a perceptron, it was made "just for fun" and it says if you will build muscles or not.

- the [hebb-perceptron](./hebb-perceptron/) has a simples example too, where the *bias* and the *hebb rule* is introducted. 

- the [classification-perceptron](./classification-perceptron/) has a more complex problem, where we test the learning capacity in 3 different datasets.

- the [classfication-adaline](./adaline/) is used in the same problem of the classification-perceptron.

in the classification problems, i wrote reports with some metrics and graphics to evaluate each classifier.

## requirements

- python3 (numpy, pandas and matplotlib)
- c++
- cmake

## references

Neural Networks and Deep Learning - Michael Nielsen [http://neuralnetworksanddeeplearning.com/]

Master's Degree Classes - Prof. Dr. Lucas C. Ribas [https://www.ibilce.unesp.br/#!/departamentos/cienc-comp-estatistica/docentes/lucas-correia-ribas/]
