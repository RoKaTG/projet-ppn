#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

//#include <stdbool.h>

#include "../matrix/matrix.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    double (*activation_func)(double);
} NeuralLayer;

double sigmoid(double x);
double sigmoid_derivative(double x);
void apply_function(Matrix* m, double (*func)(double));
void apply_function_derivative(Matrix* m, double (*func)(double));
void softmax(Matrix* m);
void softmax_derivative(Matrix* m, Matrix* expected_output);

/*************************************************************/

NeuralLayer* initialize_network(int number_of_layers, const int layer_sizes[]);

/*************************************************************/

Matrix* forward_propagation(NeuralLayer* network, int number_of_layers, Matrix* input);
void backpropagation(NeuralLayer* network, int number_of_layers, Matrix* input, Matrix* expected_output, double learning_rate);

#endif // NEURAL_NETWORK_H