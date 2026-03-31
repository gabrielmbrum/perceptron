# classification perceptron

![diagram](img/image.png)

## about

here we have 3 datasets:

- dataset1: 2 attributes | 140 training samples | 60 testing samples
- dataset2: 2 attributes | 175 training samples | 75 testing samples
- dataset3: 10 attributes | 147 training samples | 63 testing samples

different of *hebb-perceptron* we update the bias using the **hebb-rule**.

we plot important infos like the training decision boundary, test **decision boundary** and error training during the epochs

## concepts

### decision boundary

the decision boundary is a line define by: 

$$w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n - \theta = 0$$

### functions 

#### training()
- the input is the training data, the labels, the learning rate, and the number of epochs.
- by the end, it will have the updated weights, the number of completed epochs, and the error during the epochs.

#### testing()
- use the learned weights to classify new data (test set).
- calculate the accuracy.

### hyperparameters

- dataset1: epochs = 100 | learning rate = 0.1
- dataset2: epochs = 100 | learning rate = 0.1
- dataset3: epochs $\in \{100, 200, 400\}$ | learning rate $\eta \in \{0.1, 0.001, 0.0001\}$

### hebb-rule

the weights are adjusted as in hebb-perceptron, but here the bias is also adjusted as:

$$
bias_{i+1} = bias_{i} + lr * error * (-1)
$$

just to remember, the weight adjust is:

$$weight_{i + 1} = weights_{i} + lr * (expected_output - output) * input_{i}$$

the error is define by the difference between the desired output and the perceptron output.

## how to run

```
python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python3 perceptron.py
```