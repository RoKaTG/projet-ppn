#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"

// Fonctions d'activation et leurs dérivées
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x); 
}

void apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

void softmax(Matrix* m) {
    double sum = 0.0;
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] = exp(m->value[i][0]);
        sum += m->value[i][0];
    }
    for (int i = 0; i < m->row; i++) {
        m->value[i][0] /= sum;
    }
}

// Fonction pour initialiser une couche du réseau
Layer* create_layer(int input_size, int output_size, double (*activation_func)(double), double (*activation_derivative_func)(double)) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    // Initialisation des poids et des biais
    layer->weights = create_matrix(output_size, input_size);
    layer->biases = create_matrix(output_size, 1);
    if (layer->weights == NULL || layer->biases == NULL) {
        free(layer);
        return NULL;
    }

    matrix_randomize(layer->weights, 0.0, 1.0); // Random initialization
    matrix_randomize(layer->biases, 0.0, 1.0); // Random initialization

    layer->activation_function = activation_func;
    layer->activation_function_derivative = activation_derivative_func;
    
    layer->outputs = NULL;
    layer->deltas = NULL;

    return layer;
}

// Fonction pour initialiser le réseau de neurones
NeuralNetwork* create_neural_network(int* sizes, int number_of_layers, double (*activation_functions[])(double), double (*activation_derivatives[])(double)) {
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (network == NULL) {
        return NULL;
    }

    network->number_of_layers = number_of_layers;
    network->layers = (Layer**)malloc(number_of_layers * sizeof(Layer*));
    if (network->layers == NULL) {
        free(network);
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        int input_size = i == 0 ? sizes[i] : sizes[i-1];
        int output_size = sizes[i];
        network->layers[i] = create_layer(input_size, output_size, activation_functions[i], activation_derivatives[i]);
        if (network->layers[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free_layer(network->layers[j]);
            }
            free(network->layers);
            free(network);
            return NULL;
        }
    }

    return network;
}

// Fonction pour libérer une couche du réseau
void free_layer(Layer* layer) {
    if (layer != NULL) {
        free_matrix(&(layer->weights));
        free_matrix(&(layer->biases));
        free_matrix(&(layer->outputs));
        free_matrix(&(layer->deltas));
        free(layer);
    }
}

// Fonction pour libérer le réseau de neurones
void free_neural_network(NeuralNetwork* network) {
    if (network != NULL) {
        for (int i = 0; i < network->number_of_layers; i++) {
            free_layer(network->layers[i]);
        }
        free(network->layers);
        free(network);
    }
}

// Propagation avant pour une couche spécifique
void forward_propagate_layer(Layer* layer, Matrix* input) {
    if (layer == NULL || input == NULL) return;

    // Net input = Weights * Input + Biases
    Matrix* net_input = dgemm(layer->weights, input);
    add_matrix(net_input, layer->biases);
    apply_function(net_input, layer->activation_function);

    layer->outputs = net_input;
}

void forward_propagate(NeuralNetwork* network, Matrix* input) {
    if (network == NULL || input == NULL) return;

    Matrix* current_input = input;
    for (int i = 0; i < network->number_of_layers; i++) {
        forward_propagate_layer(network->layers[i], current_input);

        if (i < network->number_of_layers - 1) {
            current_input = network->layers[i]->outputs;
        }
    }
}


// Rétropropagation pour une couche spécifique
void backward_propagate_error(Layer* layer, Matrix* error, double learning_rate) {
    if (layer == NULL || error == NULL) return;

    // Gradient = Error * Derivative of the activation function
    Matrix* gradient = copy_matrix(error);
    apply_function_derivative(gradient, layer->activation_function_derivative);

    // Ajustement des poids : W += learning_rate * (Gradient * Transpose(Input))
    Matrix* input_transposed = transpose_matrix(layer->inputs);
    Matrix* delta_weights = dgemm(gradient, input_transposed);
    scale_matrix(delta_weights, learning_rate);
    add_matrix(layer->weights, delta_weights);

    // Ajustement des biais : B += learning_rate * Gradient
    scale_matrix(gradient, learning_rate);
    add_matrix(layer->biases, gradient);

    free_matrix(&delta_weights);
    free_matrix(&input_transposed);
    free_matrix(&gradient);
}

// Rétropropagation pour l'ensemble du réseau
void backward_propagate(NeuralNetwork* network, Matrix* output_error, double learning_rate) {
    if (network == NULL || output_error == NULL) return;

    Matrix* error = output_error;
    for (int i = network->number_of_layers - 1; i >= 0; i--) {
        Layer* layer = network->layers[i];
        backward_propagate_error(layer, error, learning_rate);

        if (i > 0) {
            Matrix* transposed_weights = transpose_matrix(layer->weights);
            Matrix* prev_error = dgemm(transposed_weights, error);
            free_matrix(&error);
            error = prev_error;
            free_matrix(&transposed_weights);
        }
    }

    free_matrix(&error);
}

// Fonction pour déterminer l'indice de la plus grande valeur dans un vecteur
int argmax(double* array, int length) {
    int max_index = 0;
    double max_value = array[0];
    for (int i = 1; i < length; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }
    return max_index;
}

// Fonction pour vérifier si la prédiction est correcte
bool is_correct_prediction(Matrix* output, int label) {
    int predicted_label = argmax(output->value[0], output->column);
    return predicted_label == label;
}

// Entraînement du réseau
void train_network(NeuralNetwork* network, Matrix* input_data, Matrix* output_data, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < input_data->row; i++) {
            // Sélectionner un échantillon du dataset
            Matrix* input_sample = get_row(input_data, i);
            Matrix* output_sample = get_row(output_data, i);

            // Propagation avant
            forward_propagate(network, input_sample);

            // Calcul de l'erreur
            Matrix* output_error = calculate_output_error(output_sample, network->layers[network->number_of_layers - 1]->outputs);

            // Rétropropagation
            backward_propagate(network, output_error, learning_rate);

            // Libération des ressources
            free_matrix(&input_sample);
            free_matrix(&output_sample);
            free_matrix(&output_error);
        }

        printf("Époque %d terminée.\n", epoch);
    }
}

// Évaluation du réseau
double evaluate_network(NeuralNetwork* network, Matrix* input_data, Matrix* output_data) {
    int correct_predictions = 0;
    for (int i = 0; i < input_data->row; i++) {
        Matrix* input_sample = get_row(input_data, i);
        Matrix* output_sample = get_row(output_data, i);

        forward_propagate(network, input_sample);

        if (is_correct_prediction(network->layers[network->number_of_layers - 1]->outputs, argmax(output_sample->value[0], output_sample->column))) {
            correct_predictions++;
        }

        free_matrix(&input_sample);
        free_matrix(&output_sample);
    }

    return (double)correct_predictions / input_data->row;
}

Matrix* prepare_input_data(uint8_t* images, int number_of_images) {
    Matrix* input_data = create_matrix(784, number_of_images); // 784 = 28 * 28 (taille de l'image)
    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < 784; j++) {
            input_data->value[j][i] = images[i * 784 + j] / 255.0; // Normalisation
        }
    }
    return input_data;
}

Matrix* prepare_output_data(uint8_t* labels, int number_of_images) {
    Matrix* output_data = create_matrix(10, number_of_images); // 10 pour les chiffres de 0 à 9
    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < 10; j++) {
            output_data->value[j][i] = (labels[i] == j) ? 1.0 : 0.0; // One-hot encoding
        }
    }
    return output_data;
}

Matrix* get_row(Matrix* matrix, int row_index) {
    if (row_index < 0 || row_index >= matrix->row) {
        return NULL;
    }
    Matrix* row = create_matrix(matrix->column, 1);
    for (int i = 0; i < matrix->column; i++) {
        row->value[i][0] = matrix->value[row_index][i];
    }
    return row;
}

int main() {
    srand((unsigned int)time(NULL));

    // Configuration du réseau
    int number_of_images = 100; 
    int sizes[] = {784, 128, 10}; 
    double (*activation_functions[])(double) = {sigmoid, sigmoid}; 
    double (*activation_deriv[])(double) = {sigmoid_derivative, sigmoid_derivative}; 

    NeuralNetwork* network = create_neural_network(sizes, 3, activation_functions, activation_deriv);

    // Configuration de l'entraînement
    const double learning_rate = 0.001;
    const int epochs = 10; // Nombre d'epochs pour l'entraînement

    // Chargement des données MNIST
    FILE* imageFile = fopen("../mnist/train-images-idx3-ubyte", "rb");
    FILE* labelFile = fopen("../mnist/train-labels-idx1-ubyte", "rb");
    uint8_t* images = readMnistImages(imageFile, 0, number_of_images); // Charger les images
    uint8_t* labels = readMnistLabels(labelFile, 0, number_of_images); // Charger les labels

    // Préparation des données pour l'entraînement
    Matrix* input_data = prepare_input_data(images, number_of_images);  // Convertir les images en matrices
    Matrix* output_data = prepare_output_data(labels, number_of_images); // Convertir les labels en matrices

    // Entraînement du réseau
    train_network(network, input_data, output_data, epochs, learning_rate);

    // Évaluation du réseau
    double accuracy = evaluate_network(network, input_data, output_data);
    printf("Précision du réseau sur l'ensemble d'entraînement : %.2f%%\n", accuracy * 100.0);

    // Tester avec quelques images
    for (int i = 0; i < 5; i++) {
        Matrix* input_sample = get_row(input_data, i);
        forward_propagate(network, input_sample);
        Matrix* output = network->layers[network->number_of_layers - 1]->outputs;

        printf("Image %d, Probabilités: ", i);
        for (int j = 0; j < output->row; j++) {
            printf("%.2f ", output->value[j][0]);
        }
        printf("\n");

        free_matrix(&input_sample);
    }

    // Libération des ressources
    free(images);
    free(labels);
    fclose(imageFile);
    fclose(labelFile);
    free_neural_network(network);
    free_matrix(&input_data);
    free_matrix(&output_data);

    return 0;
}

