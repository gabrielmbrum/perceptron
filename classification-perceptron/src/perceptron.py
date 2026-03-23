import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def load_data(train_db, test_db):

    df_train_loaded = pd.read_csv('../data/' + train_db)
    df_test_loaded = pd.read_csv('../data/' + test_db)

    X_train = df_train_loaded.drop('label', axis=1).values.T # (n_features, n_amostras)
    y_train = df_train_loaded['label'].values.reshape(1, -1) # (1, n_amostras)
    X_test = df_test_loaded.drop('label', axis=1).values.T
    y_test = df_test_loaded['label'].values.reshape(1, -1)

    return X_train, y_train, X_test, y_test

def dataset1() :
    print("Dataset 1:")
    X_train, y_train, X_test, y_test = load_data('train_dataset1.csv', 'test_dataset1.csv')

    perceptron = Perceptron(learning_rate=0.1, epochs=100, n_features=X_train.shape[0])

    acc_history, error_history = perceptron.train(X_train, y_train)

    accuracy = perceptron.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Accuracy on training set (last epoch): {acc_history[-1]:.2f}%')
    
    perceptron.plots(error_history, X_train, y_train, X_test, y_test, dataset_id=1)

    print("\n" + "="*50 + "\n")


def dataset2() :
    print("Dataset 2:")
    X_train, y_train, X_test, y_test = load_data('train_dataset2.csv', 'test_dataset2.csv')

    perceptron = Perceptron(learning_rate=0.1, epochs=100, n_features=X_train.shape[0])

    acc_history, error_history = perceptron.train(X_train, y_train)

    accuracy = perceptron.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Accuracy on training set (last epoch): {acc_history[-1]:.2f}%')
    
    perceptron.plots(error_history, X_train, y_train, X_test, y_test, dataset_id=2)

    print("\n" + "="*50 + "\n")

def dataset3() :
    print("Dataset 3:")
    X_train, y_train, X_test, y_test = load_data('train_dataset3.csv', 'test_dataset3.csv')

    LRs = [0.01, 0.1, 0.5]
    EPOCHS = [100, 200, 400]
    N_RUNS = 10

    for lr in LRs:
        for epoch in EPOCHS:
            print(f'\nRunning configuration: LR={lr}, Epochs={epoch}')
            
            accuracies = []
            final_error_history = None
            final_perceptron = None
            best_accuracy = -1

            for run in range(N_RUNS):
                perceptron = Perceptron(learning_rate=lr, epochs=epoch, n_features=X_train.shape[0])
                acc_history, error_history = perceptron.train(X_train, y_train)
                accuracy = perceptron.test(X_test, y_test)
                
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    final_error_history = error_history
                    final_perceptron = perceptron

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            print(f'  Mean Accuracy: {mean_acc:.2f}%')
            print(f'  Std Deviation: {std_acc:.2f}%')
            
            final_perceptron.plots(final_error_history, X_train, y_train, X_test, y_test, dataset_id=3, lr=lr, epochs=epoch)

    print("\n" + "="*50 + "\n")

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, n_features=None):
        self.learning_rate = learning_rate
        self.epochs =  epochs
        self.weights = np.random.rand(n_features)
        self.bias = random.uniform(-1, 1)

    def activation_potential(self, x):
        # u = w1*x1 + w2*x2 + ... + wn*xn - b
        return np.dot(self.weights, x) - self.bias
    
    def predict(self, X):
        y_pred = []
        for x in X.T:
            u = self.activation_potential(x)
            y_pred.append(1 if u >= 0 else -1) # degrau bipolar
        return np.array(y_pred).reshape(1, -1)

    def train(self, X, y):
        error_history = []
        acc_history = []
        for epoch in range(self.epochs):
            epoch_errors = 0
            epoch_acc = 0
            for i in range(X.shape[1]):
                x = X[:, i]
                target = y[0, i]
                
                # Predição para apenas uma amostra, extraindo o valor escalar
                y_pred_matrix = self.predict(x.reshape(-1, 1))
                y_pred = y_pred_matrix.item() # Converte array (1,1) para escalar
                
                error = target - y_pred
                
                if error != 0:
                    epoch_errors += 1 # Contabiliza erro de classificação
                else:
                    epoch_acc += 1 # Contabiliza acerto de classificação
                # Atualização dos pesos e bias
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error * (-1)
            
            error_history.append(epoch_errors)
            acc_history.append(epoch_acc / X.shape[1] * 100)
        return acc_history, error_history
    
    def test(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y) * 100
        return accuracy

    def plots(self, error_history, X_train, y_train, X_test, y_test, dataset_id, lr=None, epochs=None):
        
        output_dir = f'../output/dataset{dataset_id}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Sufixo para diferenciar os arquivos, se fornecido
        if lr is not None and epochs is not None:
            suffix = f'_lr_{lr}_epochs_{epochs}'
        else:
            suffix = ''

        # 1. Training error evolution
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(error_history) + 1), error_history, marker='o')
        plt.title(f'Training Error Evolution{f" (LR={lr}, Epochs={epochs})" if suffix else ""}')
        plt.xlabel('Epochs')
        plt.ylabel('Number of Misclassifications')
        plt.grid(True)
        plt.savefig(f'{output_dir}/training_error{suffix}.png')
        plt.close()

        if self.weights.shape[0] != 2:
            print("Decision boundary plot is only available for 2D data.")
            return

        def plot_decision_boundary(X, y, title, filename):
            plt.figure(figsize=(10, 6))
            
            class_A = np.where(y[0] == 1)[0]
            class_B = np.where(y[0] == -1)[0]
            
            plt.scatter(X[0, class_A], X[1, class_A], color='blue', label='Class 1')
            plt.scatter(X[0, class_B], X[1, class_B], color='red', label='Class -1')
            
            x1_values = np.linspace(-5, 5, 100)
            
            if self.weights[1] != 0:
                x2_values = (self.bias - self.weights[0] * x1_values) / self.weights[1]
                plt.plot(x1_values, x2_values, color='green', linewidth=2, label='Decision Boundary')
            else:
                plt.axvline(x=self.bias/self.weights[0], color='green', linewidth=2, label='Decision Boundary')
            
            plt.title(title)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.xticks(np.arange(-5, 6, 1))
            plt.yticks(np.arange(-5, 6, 1))
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_dir}/{filename}')
            plt.close()

        # 2. Training data with decision boundary
        plot_decision_boundary(X_train, y_train, f'Training Data with Decision Boundary{f" (LR={lr}, Epochs={epochs})" if suffix else ""}', f'train_decision_boundary{suffix}.png')

        # 3. Test data with decision boundary
        plot_decision_boundary(X_test, y_test, f'Test Data with Decision Boundary{f" (LR={lr}, Epochs={epochs})" if suffix else ""}', f'test_decision_boundary{suffix}.png')

if __name__ == "__main__":
    dataset1()
    dataset2()
    dataset3()