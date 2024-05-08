import matplotlib.pyplot as plt
import numpy as np

def plot_new_results(mode):
    ##
    activation_functions = ['Sigmoid', 'Fast sigmoid', 'ReLU', 'Leaky ReLU', 'Tanh', 'Swish']
    accuracies = [87.73, 91.35, 95.07, 94.58, 93.78, 94.44]
    times = [75.76, 71.32, 70.46, 70.71, 75.09, 75.59]
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    ##
    plt.figure(figsize=(10, 6))
    if mode == 0:
        plt.bar(activation_functions, accuracies, color=colors)
        plt.ylabel('Accuracy (%)')
        plt.title('Mean Accuracy in 15 Executions & 10 Epochs with mini batch of size 16')
        plt.ylim([85, 100])  
        plt.yticks(np.arange(85, 101, 1))  
        file_name = 'minibatch_a.png'
    ##
    plt.xlabel('Activation Functions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name)

# Plot for mode 0 (Accuracy Execution)
plot_new_results(0)

