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

void forward_propagate_layer(Layer* layer, Matrix* input, int layer_index, int total_layers) {
    if (layer == NULL || input == NULL) return;

    // Enregistrement de l'input pour la rétropropagation
    if (layer->inputs != NULL) {
        free_matrix(&(layer->inputs));
    }
    layer->inputs = copy_matrix(input);

////
    printf("Forward Propagate Layer: Input %dx%d, Weights %dx%d\n", 
       input->row, input->column, 
       layer->weights->row, layer->weights->column);
////

    // Net input = Weights * Input + Biases
    Matrix* net_input = dgemm(layer->weights, input);

////
    printf("Net Input Size: %dx%d\n", net_input->row, net_input->column);
////

    add_matrix(net_input, layer->biases);

    // Si c'est la dernière couche, appliquer softmax, sinon appliquer la fonction d'activation habituelle
    if (layer_index == total_layers - 1) {
        softmax(net_input);      // Appliquer softmax sur net_input
    } else {
        apply_function(net_input, layer->activation_function);
    }

    layer->outputs = net_input; // Assigner net_input à layer->outputs
////    
    printf("Forward Layer %d: Taille input %dx%d, Taille output %dx%d\n", layer_index, input->row, input->column, layer->outputs->row, layer->outputs->column);
////
}

int main() {
    return 0;
}
