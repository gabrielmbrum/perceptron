# ADALINE (Adaptive Linear Element)

This repository contains the implementation of an **ADALINE** neural network, using the **Delta Rule** for weight adjustment and minimization of the Mean Squared Error (MSE). The script allows execution in both **Stochastic** and **Batch** modes.

## Architecture Diagram

The following diagram illustrates the ADALINE architecture implemented here:

![alt text](img/image.png)

## Delta Rule

The ADALINE uses the Medium Square Error to know when the weights should stop to be recalculated.

The MSE is a squared function, so it has a minimal point and the adaline aims to reach there.

To reach the minimal point, nós usamos a decida do gradiente como guida. O gradiente é calculado pela derivada da função de MSE, então invertemos o vetor para saber a direção do ponto mínio da parábola da função.

Então, a variação de peso é definida por

$$
\Delta w = - lr \cdot \nabla E(w) 
$$

Tal que E(W) é a função de erro dos pesos w.

Expandindo a função E(w), chegamos na definição:

$$
w_{atual} = w_{anterior} + lr \sum_{k=1}^{p}(d^{(k)} - u) \cdot x^{(k)}
$$

Sendo u a função de ativação do adaline

## How to Run

The `adaline.py` script processes data files and generates training and testing results.

```bash
python3 adaline.py
```