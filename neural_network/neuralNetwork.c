#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
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

Layer* create_layer(int input_size, int output_size, double (*activation_func)(double), double (*activation_derivative_func)(double)) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    if (layer == NULL) {
        return NULL;
    }

    // Initialisation des poids et des biais
    layer->weights = create_matrix(output_size, input_size); //vecteur entrée * taille sortie couche
    layer->biases = create_matrix(output_size, 1);

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

void free_neural_network(NeuralNetwork* network) {
    if (network != NULL) {
        for (int i = 0; i < network->number_of_layers; i++) {
            free_layer(network->layers[i]);
        }
        free(network->layers);
        free(network);
    }
}

void free_layer(Layer* layer) {
    if (layer != NULL) {
        free_matrix(&(layer->weights));
        free_matrix(&(layer->biases));
        free_matrix(&(layer->outputs));
        free_matrix(&(layer->deltas));
        free(layer);
    }
}

void backward_propagate_error(Layer* layer, Matrix* error, double learning_rate) {
    if (layer == NULL || error == NULL) return;

    // Gradient = Error * Derivative of the activation function
    Matrix* gradient = copy_matrix(error);
    apply_function_derivative(gradient, layer->activation_function_derivative);

    // Ajustement des poids : W += learning_rate * (Gradient * Transpose(Input))
    Matrix* input_transposed = transpose_matrix(layer->inputs);
    Matrix* delta_weights = dgemm(gradient, input_transposed);

////
    printf("Backward Layer: Error %dx%d, Weights %dx%d, Delta Weights %dx%d\n", 
       error->row, error->column, 
       layer->weights->row, layer->weights->column, 
       delta_weights->row, delta_weights->column);
////

    scale_matrix(delta_weights, learning_rate);
    add_matrix(layer->weights, delta_weights);

    // Ajustement des biais : B += learning_rate * Gradient
    scale_matrix(gradient, learning_rate);
    add_matrix(layer->biases, gradient);

////
    printf("Backward Layer: Taille error %dx%d, Taille weights %dx%d\n", error->row, error->column, layer->weights->row, layer->weights->column);
////    
    free_matrix(&delta_weights);
    free_matrix(&input_transposed);
    free_matrix(&gradient);
}

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

int main() {
    return 0;
}
