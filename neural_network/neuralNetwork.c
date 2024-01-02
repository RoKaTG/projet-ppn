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

void forward_propagate(NeuralNetwork* network, Matrix* input) {
    if (network == NULL || input == NULL) return;

    Matrix* current_input = input;
    for (int i = 0; i < network->number_of_layers; i++) {
        forward_propagate_layer(network->layers[i], current_input, i, network->number_of_layers);

        if (i < network->number_of_layers - 1) {
            current_input = network->layers[i]->outputs;
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

int main() {
    return 0;
}
