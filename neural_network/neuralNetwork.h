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

void apply_function(Matrix* m, double (*func)(double));

#endif // NEURAL_NETWORK_H

