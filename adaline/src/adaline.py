import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Configuração para evitar warnings de interface gráfica em execuções batch
plt.switch_backend('Agg') 

def load_data(train_db, test_db):
    df_train_loaded = pd.read_csv('../data/' + train_db)
    df_test_loaded = pd.read_csv('../data/' + test_db)

    X_train = df_train_loaded.drop('label', axis=1).values.T # (n_features, n_amostras)
    y_train = df_train_loaded['label'].values.reshape(1, -1) # (1, n_amostras)
    X_test = df_test_loaded.drop('label', axis=1).values.T
    y_test = df_test_loaded['label'].values.reshape(1, -1)

    return X_train, y_train, X_test, y_test

def dataset1(metrics_log):
    print("Dataset 1:")
    X_train, y_train, X_test, y_test = load_data('train_dataset1.csv', 'test_dataset1.csv')

    print("STOCHASTIC GRADIENT DESCENT:")
    adaline = Adaline(learning_rate=0.01, max_epochs=100, n_features=X_train.shape[0], precision=0.0001, gradient='stochastic')
    mse_history, cost_history = adaline.train(X_train, y_train)
    # Exporta CSV e gera gráficos automaticamente (MSE, Class Error, Boundaries)
    summary = adaline.export_and_plot(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id=1)
    metrics_log.append(summary)

    print("\nBATCH GRADIENT DESCENT:")
    adaline = Adaline(learning_rate=0.01, max_epochs=100, n_features=X_train.shape[0], precision=0.0001, gradient='batch')
    mse_history, cost_history = adaline.train(X_train, y_train)
    summary = adaline.export_and_plot(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id=1)
    metrics_log.append(summary)
    
    print("\n" + "="*50 + "\n")

def dataset2(metrics_log):
    print("Dataset 2:")
    X_train, y_train, X_test, y_test = load_data('train_dataset2.csv', 'test_dataset2.csv')

    print("STOCHASTIC GRADIENT DESCENT:")
    adaline = Adaline(learning_rate=0.01, max_epochs=100, n_features=X_train.shape[0], precision=0.0001, gradient='stochastic')
    mse_history, cost_history = adaline.train(X_train, y_train)
    summary = adaline.export_and_plot(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id=2)
    metrics_log.append(summary)

    print("\nBATCH GRADIENT DESCENT:")
    adaline = Adaline(learning_rate=0.01, max_epochs=100, n_features=X_train.shape[0], precision=0.0001, gradient='batch')
    mse_history, cost_history = adaline.train(X_train, y_train)
    summary = adaline.export_and_plot(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id=2)
    metrics_log.append(summary)

    print("\n" + "="*50 + "\n")

def dataset3(metrics_log):
    print("Dataset 3:")
    X_train, y_train, X_test, y_test = load_data('train_dataset3.csv', 'test_dataset3.csv')

    LRs = [0.01, 0.001, 0.0001]
    epoch = 100
    PRECISIONS = [0.1, 0.0001]
    GRADIENTS = ['stochastic', 'batch']

    for gradient in GRADIENTS:
        for lr in LRs:
            for precision in PRECISIONS:
                print(f'\nRunning configuration: LR={lr}, Precision={precision}, Gradient={gradient}')
                
                adaline = Adaline(learning_rate=lr, max_epochs=epoch, n_features=X_train.shape[0], precision=precision, gradient=gradient)
                mse_history, cost_history = adaline.train(X_train, y_train)
                # Exporta CSV e gera gráfico automaticamente (Apenas MSE para DS3)
                summary = adaline.export_and_plot(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id=3)
                metrics_log.append(summary)

    print("\n" + "="*50 + "\n")


class Adaline:
    def __init__(self, learning_rate=0.01, max_epochs=1000, n_features=None, precision=0.0001, gradient='stochastic'):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        # Adicionada seed fixa para reprodutibilidade nos relatórios
        np.random.seed(42) 
        self.weights = np.random.rand(n_features + 1)
        self.last_epoch = 0
        self.precision = precision
        self.gradient = gradient

    def activation_potential(self, x):
        return np.dot(self.weights, x)
    
    def predict(self, X):
        X_bias = np.vstack([-np.ones((1, X.shape[1])), X])
        y_pred = []
        for x in X_bias.T:
            u = self.activation_potential(x)
            y_pred.append(1 if u >= 0 else -1)
        return np.array(y_pred).reshape(1, -1)

    def train(self, X, y):
        X_bias = np.vstack([-np.ones((1, X.shape[1])), X])
        
        mse_history = []
        cost_history = []

        for epoch in range(self.max_epochs):
            epoch_errors = 0
            epoch_error_sum = np.zeros(self.weights.shape)
            eqm_atual = 0

            if epoch > 0:
                eqm_anterior = eqm_atual

            for i in range(X_bias.shape[1]):
                x = X_bias[:, i] 
                target = y[0, i]
                
                u = self.activation_potential(x)
                y_pred_scalar = 1 if u >= 0 else -1
                
                error = target - u

                if y_pred_scalar != target:
                    epoch_errors += 1

                if self.gradient == 'stochastic':
                    self.weights += self.learning_rate * error * x
                elif self.gradient == 'batch':
                    epoch_error_sum += error * x    

            if self.gradient == 'batch':
                self.weights += self.learning_rate * epoch_error_sum * (1 / X_bias.shape[1])
            
            u_all = np.dot(self.weights, X_bias)
            eqm_atual = np.mean((y - u_all) ** 2)
            mse_history.append(eqm_atual)

            cost_history.append(epoch_errors / X_bias.shape[1])

            if epoch > 0 and abs(eqm_atual - eqm_anterior) <= self.precision:
                print(f'Convergence reached at epoch {epoch + 1}. eqm: {eqm_atual:.6f}')
                break
        
        self.last_epoch = epoch + 1
        return mse_history, cost_history
    
    def test(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y) * 100
        return accuracy

    def export_and_plot(self, mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id):
        """Metodo mestre que exporta dados e chama a geracao de graficos requisitados."""
        output_dir = f'../output/dataset{dataset_id}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Prefixo comum para arquivos (CSV e Plots) baseado nos hiperparametros
        suffix = f'_lr_{self.learning_rate}_prec_{self.precision}_{self.gradient}' if dataset_id == 3 else f'_{self.gradient}' 
        
        # --- 1. Exportar Historico para CSV ---
        df_history = pd.DataFrame({
            'Epoch': range(1, len(mse_history) + 1),
            'MSE': mse_history,
            'Classification_Error': cost_history
        })
        history_filename = f'history{suffix}.csv'
        df_history.to_csv(os.path.join(output_dir, history_filename), index=False)
        
        # Calcula acurácias finais para o summary
        train_acc = (1 - cost_history[-1]) * 100
        test_acc = self.test(X_test, y_test)
        
        print(f'  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
        print(f'  Exported history to {history_filename}')

        # --- 2. Gerar Graficos Requisitados pela Atividade ---
        self.generate_plots(mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id, suffix)

        # Retorna um dicionário com o resumo para o Log Mestre (summary_metrics.csv)
        return {
            'Dataset': dataset_id,
            'Gradient_Mode': self.gradient,
            'Learning_Rate': self.learning_rate,
            'Precision': self.precision,
            'Epochs_Run': self.last_epoch,
            'Final_MSE': mse_history[-1],
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Weights': str(np.round(self.weights, 4).tolist())
        }

    def generate_plots(self, mse_history, cost_history, X_train, y_train, X_test, y_test, dataset_id, suffix):
        """Gera graficos baseados nos requisitos especificos de cada dataset."""
        # Pasta especifica para plots dentro do dataset
        plots_dir = f'../output/dataset{dataset_id}/plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        epochs_range = range(1, len(mse_history) + 1)
        
        # Título base para os gráficos
        title_base = f'DS{dataset_id} | LR={self.learning_rate} | Prec={self.precision} | Mode={self.gradient}'

        # --- REQUISITO COMUM (DS 1, 2, 3): Evolução do EQM (MSE) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs_range, mse_history, marker='o', linestyle='-', color='b', label='MSE (Eqm)')
        ax.set_title(f'MSE Evolution\n{title_base}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Squared Error')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Força eixo X inteiro
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(plots_dir, f'mse_evolution{suffix}.png'))
        plt.close()

        # --- REQUISITOS ESPECIFICOS PARA DATASET 1 e 2 ---
        if dataset_id in [1, 2]:
            # 1. Gráfico da evolução do erro de classificação
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_range, cost_history, marker='o', linestyle='-', color='orange', label='Class. Error')
            ax.set_title(f'Classification Error Evolution\n{title_base}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Classification Error Rate')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
            ax.grid(True)
            plt.savefig(os.path.join(plots_dir, f'classification_error{suffix}.png'))
            plt.close()

            # 2. Fronteiras de Decisão (Apenas se for 2D, o que DS1 e DS2 sao)
            if X_train.shape[0] == 2:
                # Fronteira nos dados de Treinamento
                self._plot_boundary(X_train, y_train, 
                                   title=f'Training Data - Decision Boundary\n{title_base}', 
                                   filepath=os.path.join(plots_dir, f'boundary_train{suffix}.png'))
                
                # Fronteira nos dados de Teste (usando a mesma fronteira aprendida)
                self._plot_boundary(X_test, y_test, 
                                   title=f'Test Data - Decision Boundary\n{title_base}', 
                                   filepath=os.path.join(plots_dir, f'boundary_test{suffix}.png'))

    def _plot_boundary(self, X, y, title, filepath):
        """Metodo auxiliar para plotar pontos 2D e a linha de fronteira."""
        plt.figure(figsize=(10, 6))
        
        # Separa os indices das classes
        class_1_indices = np.where(y[0] == 1)[0]
        class_minus_1_indices = np.where(y[0] == -1)[0]
        
        # Plota os pontos
        plt.scatter(X[0, class_1_indices], X[1, class_1_indices], color='blue', label='Class 1', alpha=0.7)
        plt.scatter(X[0, class_minus_1_indices], X[1, class_minus_1_indices], color='red', label='Class -1', alpha=0.7)
        
        # Calcula a linha da fronteira: w0 - w1*x1 - w2*x2 = 0  =>  x2 = (w0 - w1*x1) / w2
        # Define intervalo de x1 baseado nos dados
        x1_min, x1_max = X[0, :].min() - 1, X[0, :].max() + 1
        x1_values = np.linspace(x1_min, x1_max, 100)
        
        w0 = self.weights[0] # Bias é index 0
        w1 = self.weights[1]
        w2 = self.weights[2]

        if w2 != 0:
            x2_values = (w0 - w1 * x1_values) / w2
            plt.plot(x1_values, x2_values, color='green', linewidth=3, label='Decision Boundary')
        else:
            # Caso raro de w2 ser zero (divisao por zero)
            plt.axvline(x=w0/w1, color='green', linewidth=3, label='Decision Boundary')
        
        # Ajustes do gráfico baseados nos arquivos fornecidos
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        # Tenta manter os limites padroes [-5, 5] se os dados permitirem, senao ajusta automaticamente
        data_x_lims = [X[0, :].min() - 0.5, X[0, :].max() + 0.5]
        data_y_lims = [X[1, :].min() - 0.5, X[1, :].max() + 0.5]
        
        # plt.xlim(max(-5, data_x_lims[0]), min(5, data_x_lims[1])) # Tentativa de fixar -5,5
        # plt.ylim(max(-5, data_y_lims[0]), min(5, data_y_lims[1]))
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(filepath)
        plt.close()


if __name__ == "__main__":
    metrics_log = []
    
    # Cria pasta output geral se nao existir
    if not os.path.exists('../output'):
        os.makedirs('../output')

    dataset1(metrics_log)
    dataset2(metrics_log)
    dataset3(metrics_log)
    
    # Salva o log mestre com todas as execuções (atende requisito de acuracias do DS3 globalmente)
    df_metrics = pd.DataFrame(metrics_log)
    master_log_path = os.path.join('../output', 'summary_metrics.csv')
    df_metrics.to_csv(master_log_path, index=False)
    print(f"\n[SUCESSO] Log mestre de métricas salvo em: {master_log_path}")
    print("Todos os graficos requisitados foram salvos nas subpastas 'plots' dentro de '../output/datasetX/'")