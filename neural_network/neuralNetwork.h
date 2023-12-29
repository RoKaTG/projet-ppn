#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

//#include <stdbool.h>

#include "../matrix/matrix.h"

typedef struct {
    Matrix* weights;       // Matrice des poids
    Matrix* biases;        // Vecteur des biais
    Matrix* outputs;       // Sorties de la couche
    Matrix* inputs;
    Matrix* partial_derivatives;
    Matrix* deltas;        // Deltas pour la rétropropagation
    double (*activation_function)(double); // Fonction d'activation
    double (*activation_function_derivative)(double); // Dérivée de la fonction d'activation
} Layer;

typedef struct {
    int number_of_layers; // Nombre de couches
    Layer** layers;       // Tableau des couches
} NeuralNetwork;

double sigmoid(double x);
double sigmoid_derivative(double x);
void apply_function(Matrix* m, double (*func)(double));
void apply_function_derivative(Matrix* m, double (*func)(double));
void softmax(Matrix* m);

Layer* create_layer(int input_size, int output_size, double (*activation_func)(double), double (*activation_derivative_func)(double));
void free_layer(Layer* layer);

NeuralNetwork* create_neural_network(int* sizes, int number_of_layers, double (*activation_functions[])(double), double (*activation_derivatives[])(double), int firstLayerSize);
void free_neural_network(NeuralNetwork* network);

void backward_propagate_error(Layer* layer, Matrix* next_layer_weights, Matrix* next_layer_deltas);
void forward_propagate_layer(Layer* layer, Matrix* input, int layer_index, int total_layers);

Matrix* calculate_partial_derivatives(Layer* layer, Matrix* net_input);
void calculate_gradient(Layer* layer);
void adjust_weights_and_biases(Layer* layer, double learning_rate);

void backward_propagate(NeuralNetwork* network, Matrix* output_error, double learning_rate);
void forward_propagate(NeuralNetwork* network, Matrix* input);

void train_network(NeuralNetwork* network, Matrix* input_data, Matrix* output_data, int epochs, double learning_rate);
Matrix* prepare_input_data(uint8_t* images, int number_of_images);
Matrix* prepare_output_data(uint8_t* labels, int number_of_images);
Matrix* get_row(Matrix* matrix, int row_index);
Matrix* get_column(Matrix* matrix, int col_index);

Matrix* calculate_output_error(Matrix* expected_output, Matrix* actual_output);

#endif // NEURAL_NETWORK_H
