#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "../matrix/matrix.h"

typedef struct {
    Matrix* weights;       // Matrice des poids
    Matrix* biases;        // Vecteur des biais
    Matrix* outputs;       // Sorties de la couche
    Matrix* inputs;
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
void softmax(Matrix* m);

void apply_function(Matrix* m, double (*func)(double));
void apply_function_derivative(Matrix* m, double (*func)(double));

Layer* create_layer(int input_size, int output_size, double (*activation_func)(double), double (*activation_derivative_func)(double));
NeuralNetwork* create_neural_network(int* sizes, int number_of_layers, double (*activation_functions[])(double), double (*activation_derivatives[])(double), int firstLayerSize);

void free_neural_network(NeuralNetwork* network);
void free_layer(Layer* layer);

void forward_propagate_layer(Layer* layer, Matrix* input, int layer_index, int total_layers);

#endif // NEURAL_NETWORK_H

