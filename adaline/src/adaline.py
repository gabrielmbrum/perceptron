import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(train_db, test_db):
    df_train_loaded = pd.read_csv('../data/' + train_db)
    df_test_loaded = pd.read_csv('../data/' + test_db)

    X_train = df_train_loaded.drop('label', axis=1).values.T # (n_features, n_amostras)
    y_train = df_train_loaded['label'].values.reshape(1, -1) # (1, n_amostras)
    X_test = df_test_loaded.drop('label', axis=1).values.T
    y_test = df_test_loaded['label'].values.reshape(1, -1)

    return X_train, y_train, X_test, y_test

def dataset1():
    print("Dataset 1 (Training with SGD):")
    X_train, y_train, X_test, y_test = load_data('train_dataset1.csv', 'test_dataset1.csv')

    adaline = Adaline(learning_rate=0.01, epochs=100, precision=0.001, n_features=X_train.shape[0])
    mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=1, method='sgd')

    accuracy = adaline.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Misclassifications on training set (last epoch): {class_error_history[-1]}')
    
    adaline.plots(mse_history, class_error_history, X_train, y_train, X_test, y_test, dataset_id=1, suffix='_sgd')

    print("\n" + "-"*25 + "\n")

    mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=1, method='batch')

    accuracy = adaline.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Misclassifications on training set (last epoch): {class_error_history[-1]}')
    
    adaline.plots(mse_history, class_error_history, X_train, y_train, X_test, y_test, dataset_id=1, suffix='_batch')

    print("\n" + "="*50 + "\n")

def dataset2():
    print("Dataset 2 (Training with Batch):")
    X_train, y_train, X_test, y_test = load_data('train_dataset2.csv', 'test_dataset2.csv')

    # Nota: LRs para Batch Gradient Descent na soma dos erros tendem a precisar ser muito menores
    adaline = Adaline(learning_rate=0.001, epochs=100, precision=0.001, n_features=X_train.shape[0])
    mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=2, method='batch')

    accuracy = adaline.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Misclassifications on training set (last epoch): {class_error_history[-1]}')
    
    adaline.plots(mse_history, class_error_history, X_train, y_train, X_test, y_test, dataset_id=2, suffix='_batch')

    print("\n" + "-"*25 + "\n")

    mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=2, method='sgd')

    accuracy = adaline.test(X_test, y_test)
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Misclassifications on training set (last epoch): {class_error_history[-1]}')
    
    adaline.plots(mse_history, class_error_history, X_train, y_train, X_test, y_test, dataset_id=2, suffix='_sgd')

    print("\n" + "="*50 + "\n")

def dataset3():
    print("Dataset 3 (Evaluating SGD Configurations):")
    X_train, y_train, X_test, y_test = load_data('train_dataset3.csv', 'test_dataset3.csv')

    LRs = [0.01, 0.001, 0.0001] 
    PRECISIONS = [0.1, 0.0001]
    N_RUNS = 10
    EPOCH = 100

    for lr in LRs:
        for precision in PRECISIONS:
            print(f'\nRunning configuration: LR={lr}, Precision={precision}')
            
            accuracies = []
            final_mse_history = None
            final_class_error_history = None
            final_adaline = None
            best_accuracy = -1

            for run in range(N_RUNS):
                adaline = Adaline(learning_rate=lr, epochs=EPOCH, precision=precision, n_features=X_train.shape[0])
                suffix = f'_lr_{lr}_precision_{precision}_run_{run}'
                mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=3, method='sgd', suffix=suffix)
                
                accuracy = adaline.test(X_test, y_test)
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    final_mse_history = mse_history
                    final_class_error_history = class_error_history
                    final_adaline = adaline

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            print(f'  Mean Accuracy: {mean_acc:.2f}%')
            print(f'  Std Deviation: {std_acc:.2f}%')
            
            best_suffix = f'_lr_{lr}_precision_{precision}_best'
            final_adaline.plots(final_mse_history, final_class_error_history, X_train, y_train, X_test, y_test, dataset_id=3, suffix=best_suffix)

    print("\n" + "-"*25 + "\n")

    for lr in LRs:
        for precision in PRECISIONS:
            print(f'\nRunning configuration: LR={lr}, Precision={precision}')
            
            accuracies = []
            final_mse_history = None
            final_class_error_history = None
            final_adaline = None
            best_accuracy = -1

            for run in range(N_RUNS):
                adaline = Adaline(learning_rate=lr, epochs=EPOCH, precision=precision, n_features=X_train.shape[0])
                suffix = f'_lr_{lr}_precision_{precision}_run_{run}'
                mse_history, class_error_history = adaline.train(X_train, y_train, dataset_id=3, method='batch', suffix=suffix)
                
                accuracy = adaline.test(X_test, y_test)
                accuracies.append(accuracy)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    final_mse_history = mse_history
                    final_class_error_history = class_error_history
                    final_adaline = adaline

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            print(f'  Mean Accuracy: {mean_acc:.2f}%')
            print(f'  Std Deviation: {std_acc:.2f}%')
            
            best_suffix = f'_lr_{lr}_precision_{precision}_best'
            final_adaline.plots(final_mse_history, final_class_error_history, X_train, y_train, X_test, y_test, dataset_id=3, suffix=best_suffix)

    print("\n" + "="*50 + "\n")

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=1000, precision=1e-5, n_features=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.precision = precision
        # Inicializa pesos com (n_features + 1) para acomodar o w0 (bias) no índice 0
        np.random.seed(42) 
        self.weights = np.random.rand(n_features + 1)

    def _add_bias_term(self, X):
        """
        Adiciona uma linha de 1s no topo (index 0) da matriz de entrada X.
        Se X tem shape (n_features, n_samples), o retorno terá (n_features + 1, n_samples).
        """
        ones = np.ones((1, X.shape[1]))
        return np.vstack((ones, X))

    def activation_potential(self, X_biased):
        # O bias (w0) já está contido em self.weights, e o X_biased já possui x0 = 1
        return np.dot(self.weights, X_biased)
    
    def predict(self, X):
        X_biased = self._add_bias_term(X)
        u = self.activation_potential(X_biased)
        return np.where(u >= 0, 1, -1)

    def train_stochastic_gradient_descent(self, X_biased, y, log_file):
        mse_history = []
        class_error_history = []
        previous_mse = float('inf')
        n_samples = X_biased.shape[1]

        with open(log_file, 'w') as f:
            f.write("epoch,mse,classification_error\n")
            
            for epoch in range(self.epochs):
                for i in range(n_samples):
                    x_i = X_biased[:, i]
                    target_i = y[0, i]
                    
                    u_i = self.activation_potential(x_i)
                    error = target_i - u_i
                    
                    # Atualiza todos os pesos de uma vez, incluindo w0
                    self.weights += self.learning_rate * error * x_i
                
                # Métricas ao final da época
                u_all = self.activation_potential(X_biased)
                mse = np.mean((y - u_all) ** 2)
                
                # Para classificação, u >= 0 vira 1, caso contrário -1
                y_pred = np.where(u_all >= 0, 1, -1)
                misclassifications = np.sum(y_pred != y)
                
                mse_history.append(mse)
                class_error_history.append(misclassifications)
                f.write(f"{epoch + 1},{mse},{misclassifications}\n")
                
                if abs(mse - previous_mse) <= self.precision:
                    print(f'SGD Training stopped early at epoch {epoch + 1} due to precision criterion.')
                    break
                previous_mse = mse
                
        return mse_history, class_error_history

    def train_batch_gradient_descent(self, X_biased, y, log_file):
        mse_history = []
        class_error_history = []
        previous_mse = float('inf')
        
        with open(log_file, 'w') as f:
            f.write("epoch,mse,classification_error\n")
            
            for epoch in range(self.epochs):
                # O Erro no Batch é o vetor com os erros de todas as amostras
                u_all = self.activation_potential(X_biased)
                errors = y - u_all # Shape (1, n_samples)
                
                # O gradiente é a soma dos produtos (erro * x) para todas as amostras
                gradient = np.dot(errors, X_biased.T).flatten()
                
                # Atualização única por época
                self.weights += self.learning_rate * gradient

                # Métricas
                mse = np.mean(errors ** 2)
                y_pred = np.where(u_all >= 0, 1, -1)
                misclassifications = np.sum(y_pred != y)
                
                mse_history.append(mse)
                class_error_history.append(misclassifications)
                f.write(f"{epoch + 1},{mse},{misclassifications}\n")
                
                if abs(mse - previous_mse) <= self.precision:
                    print(f'Batch Training stopped early at epoch {epoch + 1} due to precision criterion.')
                    break
                previous_mse = mse
                
        return mse_history, class_error_history

    def train(self, X, y, dataset_id, method='sgd', suffix=''):
        output_dir = f'../output/dataset{dataset_id}'
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f'training_log{suffix}_{method}.csv')
        
        # Adiciona a linha de '1's para o bias antes de repassar aos loops de treinamento
        X_biased = self._add_bias_term(X)
        
        if method == 'sgd':
            return self.train_stochastic_gradient_descent(X_biased, y, log_file)
        elif method == 'batch':
            return self.train_batch_gradient_descent(X_biased, y, log_file)
        else:
            raise ValueError("Training method must be 'sgd' or 'batch'")
    
    def test(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y) * 100
        return accuracy

    def plots(self, mse_history, class_error_history, X_train, y_train, X_test, y_test, dataset_id, suffix=''):
        output_dir = f'../output/dataset{dataset_id}'
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mse_history) + 1), mse_history, color='purple', marker='o', markersize=4)
        plt.title(f'MSE Evolution over Epochs {suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.grid(True)
        plt.savefig(f'{output_dir}/mse_evolution{suffix}.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(class_error_history) + 1), class_error_history, color='orange', marker='o', markersize=4)
        plt.title(f'Classification Error Evolution {suffix}')
        plt.xlabel('Epochs')
        plt.ylabel('Number of Misclassifications')
        plt.grid(True)
        plt.savefig(f'{output_dir}/classification_error{suffix}.png')
        plt.close()

        # O self.weights agora tem shape n_features + 1. Para 2D, precisa ter 3 valores (w0, w1, w2)
        if self.weights.shape[0] != 3:
            print("Decision boundary plot is only available for 2D data.")
            return

        def plot_decision_boundary(X, y, title, filename):
            plt.figure(figsize=(10, 6))
            
            class_A = np.where(y[0] == 1)[0]
            class_B = np.where(y[0] == -1)[0]
            
            # X original é recebido aqui (sem a linha de 1s), então X[0] é a feature 1, X[1] é a feature 2
            plt.scatter(X[0, class_A], X[1, class_A], color='blue', label='Class 1')
            plt.scatter(X[0, class_B], X[1, class_B], color='red', label='Class -1')
            
            x1_values = np.linspace(-5, 5, 100)
            
            w0, w1, w2 = self.weights[0], self.weights[1], self.weights[2]
            
            # Considerando u = w0*1 + w1*x1 + w2*x2 = 0 => x2 = (-w0 - w1*x1) / w2
            if w2 != 0:
                x2_values = (-w0 - w1 * x1_values) / w2
                plt.plot(x1_values, x2_values, color='green', linewidth=2, label='Decision Boundary')
            else:
                plt.axvline(x=-w0/w1, color='green', linewidth=2, label='Decision Boundary')
            
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

        plot_decision_boundary(X_train, y_train, f'Training Data with Decision Boundary {suffix}', f'train_decision_boundary{suffix}.png')
        plot_decision_boundary(X_test, y_test, f'Test Data with Decision Boundary {suffix}', f'test_decision_boundary{suffix}.png')

if __name__ == "__main__":
    dataset1()
    dataset2()
    dataset3()