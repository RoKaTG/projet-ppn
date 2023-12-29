#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"

// Fonctions d'activation et leurs dérivées
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

void apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

void apply_function_derivative(Matrix* m, double (*func)(double)) {
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
    layer->weights = create_matrix(output_size, input_size); //vecteur entrée * taille sortie couche
    layer->biases = create_matrix(output_size, 1);
    layer->partial_derivatives = create_matrix(output_size, input_size);

////
    printf("Layer created: Weights %dx%d, Biases %dx%d\n", 
       layer->weights->row, layer->weights->column, 
       layer->biases->row, layer->biases->column);
////

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
NeuralNetwork* create_neural_network(int* sizes, int number_of_layers, double (*activation_functions[])(double), double (*activation_derivatives[])(double), int firstLayerSize) {
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (network == NULL) {
        return NULL;
    }

    firstLayerSize = 784;

    network->number_of_layers = number_of_layers;
    network->layers = (Layer**)malloc(number_of_layers * sizeof(Layer*));
    if (network->layers == NULL) {
        free(network);
        return NULL;
    }

    for (int i = 0; i < number_of_layers; i++) {
        int input_size = i == 0 ? firstLayerSize : sizes[i-1];
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
        free_matrix(&(layer->partial_derivatives));
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

















Matrix* calculate_partial_derivatives(Layer* layer, Matrix* net_input) {
    // Calculate the partial derivative
    Matrix* derivatives = create_matrix(net_input->row, net_input->column);
    
    for (int i = 0; i < derivatives->row; i++) {
        for (int j = 0; j < derivatives->column; j++) {
            derivatives->value[i][j] = layer->activation_function_derivative(net_input->value[i][j]);
        }
    }

    return derivatives;
}

// Assuming error is already calculated and stored in layer->deltas
void calculate_gradient(Layer* layer) {
    for (int i = 0; i < layer->deltas->row; i++) {
        for (int j = 0; j < layer->deltas->column; j++) {
            layer->deltas->value[i][j] *= layer->activation_function_derivative(layer->outputs->value[i][j]);
        }
    }
}

void adjust_weights_and_biases(Layer* layer, double learning_rate) {
    Matrix* transposed_inputs = transpose_matrix(layer->inputs);
    Matrix* weight_adjustment = dgemm(layer->deltas, transposed_inputs);

    scale_matrix(weight_adjustment, learning_rate);
    add_matrix(layer->weights, weight_adjustment);

    for (int i = 0; i < layer->biases->row; i++) {
        for (int j = 0; j < layer->biases->column; j++) {
            layer->biases->value[i][j] += learning_rate * layer->deltas->value[i][j];
        }
    }

    free_matrix(&transposed_inputs);
    free_matrix(&weight_adjustment);
}

void forward_propagate_layer(Layer* layer, Matrix* input, int layer_index, int total_layers) {
    if (layer == NULL || input == NULL) return;

    // Store the input for the backward propagation
    if (layer->inputs != NULL) {
        free_matrix(&(layer->inputs));
    }
    layer->inputs = copy_matrix(input);

    // Net input = Weights * Input + Biases
    Matrix* net_input = dgemm(layer->weights, input);
    add_matrix(net_input, layer->biases);

    // Apply activation function & calculate the partials derivatives
    apply_function(net_input, layer->activation_function);

    // Calculate partials derivatives for the backward propagation
    layer->partial_derivatives = calculate_partial_derivatives(layer, net_input);

    layer->outputs = net_input;
}

void backward_propagate_error(Layer* layer, Matrix* next_layer_weights, Matrix* next_layer_deltas) {
    if (layer == NULL || next_layer_weights == NULL || next_layer_deltas == NULL) return;

    // Calculate the error for the current layer
    Matrix* transposed_weights = transpose_matrix(next_layer_weights);
    Matrix* error = dgemm(transposed_weights, next_layer_deltas);
    free_matrix(&transposed_weights);

    // Store the error in layer->deltas for gradient calculation
    if (layer->deltas != NULL) free_matrix(&(layer->deltas));
    layer->deltas = error;

    calculate_gradient(layer);
}

void backward_propagate(NeuralNetwork* network, Matrix* output_error, double learning_rate) {
    if (network == NULL || output_error == NULL) return;

    Matrix* error = output_error;
    for (int i = network->number_of_layers - 1; i > 0; i--) {
        Layer* current_layer = network->layers[i];
        Layer* previous_layer = network->layers[i - 1];

        backward_propagate_error(previous_layer, current_layer->weights, error);
        free_matrix(&error);

        if (i > 1) { // Only calculate error for layers before the last hidden layer
            error = copy_matrix(previous_layer->deltas);
        }

        adjust_weights_and_biases(current_layer, learning_rate);
    }

    // For the first layer, we don't need to calculate the error
    adjust_weights_and_biases(network->layers[0], learning_rate);
}

void forward_propagate(NeuralNetwork* network, Matrix* input) {
    if (network == NULL || input == NULL) return;

    Matrix* current_input = input;
    for (int i = 0; i < network->number_of_layers; i++) {
        forward_propagate_layer(network->layers[i], current_input, i, network->number_of_layers);

        if (i < network->number_of_layers - 1) {
            current_input = network->layers[i]->outputs;
        }
    }
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

////
Matrix* get_column(Matrix* matrix, int col_index) {
    if (col_index < 0 || col_index >= matrix->column) {
        return NULL;
    }
    Matrix* column = create_matrix(matrix->row, 1);
    for (int i = 0; i < matrix->row; i++) {
        column->value[i][0] = matrix->value[i][col_index];
    }
    return column;
}
////

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

// Entraînement du réseau
void train_network(NeuralNetwork* network, Matrix* input_data, Matrix* output_data, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {

////
        printf("Epoch %d start: input_data size %dx%d, output_data size %dx%d\n",
                epoch, input_data->row, input_data->column, output_data->row, output_data->column);
////

        for (int i = 0; i < input_data->row; i++) {
            // Sélectionner un échantillon du dataset
            
////
            Matrix* input_sample = get_column(input_data, i);
////

            //Matrix* input_sample = get_row(input_data, i);
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

//// 
        printf("Epoch %d end: input_data size %dx%d, output_data size %dx%d\n",
               epoch, input_data->row, input_data->column, output_data->row, output_data->column);
////
    }
}

Matrix* calculate_output_error(Matrix* expected_output, Matrix* actual_output) {
    if (expected_output == NULL || actual_output == NULL ||
        expected_output->row != actual_output->row || expected_output->column != actual_output->column) {
        return NULL; // Gestion d'erreur
    }

    Matrix* error = create_matrix(expected_output->row, expected_output->column);
    if (error == NULL) {
        return NULL; // Gestion d'erreur si la création de la matrice échoue
    }

    for (int i = 0; i < error->row; i++) {
        for (int j = 0; j < error->column; j++) {
            double diff = expected_output->value[i][j] - actual_output->value[i][j];
            error->value[i][j] = diff * diff; // Calcul de l'erreur quadratique
        }
    }

    return error;
}

int main() {
    srand((unsigned int)time(NULL));

    printf("Ceci est le PROJET PPN2\n");

    // Configuration du réseau
    int number_of_images = 1000; 
    int sizes[] = {128, 10}; 
    double (*activation_functions[])(double) = {sigmoid, sigmoid}; 
    double (*activation_deriv[])(double) = {sigmoid_derivative, sigmoid_derivative}; 

    NeuralNetwork* network = create_neural_network(sizes, 2, activation_functions, activation_deriv, 784);

////
    printf("Réseau initialisé avec %d couches\n", network->number_of_layers);

////
    // Configuration de l'entraînement
    const double learning_rate = 0.001;
    const int epochs = 100; // Nombre d'epochs pour l'entraînement

    // Chargement des données MNIST
    FILE* imageFile = fopen("../mnist/train-images-idx3-ubyte", "rb");
    FILE* labelFile = fopen("../mnist/train-labels-idx1-ubyte", "rb");
    uint8_t* images = readMnistImages(imageFile, 0, number_of_images); // Charger les images
    uint8_t* labels = readMnistLabels(labelFile, 0, number_of_images); // Charger les labels

    // Préparation des données pour l'entraînement
    Matrix* input_data = prepare_input_data(images, number_of_images);  // Convertir les images en matrices
    Matrix* output_data = prepare_output_data(labels, number_of_images); // Convertir les labels en matrices

    // Entraînement du réseau
 
////
     printf("Before training: input_data size %dx%d, output_data size %dx%d\n",
           input_data->row, input_data->column, output_data->row, output_data->column);
////

    train_network(network, input_data, output_data, epochs, learning_rate);

////
    printf("After training: input_data size %dx%d, output_data size %dx%d\n",
           input_data->row, input_data->column, output_data->row, output_data->column);
////

    // Tester avec quelques images
    for (int i = 0; i < 200; i++) {
        Matrix* input_sample = get_column(input_data, i);
        printf("Taille de input_data : %d * %d\n", input_data->row, input_data->column);
        printf("Taille de input_sample : %d * %d\n", input_sample->row, input_sample->column);
        forward_propagate(network, input_sample);
        Matrix* output = network->layers[network->number_of_layers - 1]->outputs;

        printf("Image %d, Labels %d  Probabilités: ", i, labels[i]);
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
