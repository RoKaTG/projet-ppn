#include <cblas.h>
#include <clapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

#include "../../include/networks/mlp_blas.h"
#include "../../include/networks/activation.h"
 
/**************************************/
/*                                    */
/*          Network creation          */
/*                                    */
/**************************************/

/**
 * Creates a multilayer perceptron (MLP) neural network with a specified structure and parameters.
 * Initializes the layers, weights, biases, and learning rate.
 * Weights are randomly initialized, and biases are initialized to zero.
 *
 * @param numLayers The number of layers in the network.
 * @param layerSizes An array containing the sizes of each layer.
 * @param learningRate The learning rate for weight updates during training.
 * @return A pointer to the newly created MLP network.
 */
MLP* create_mlp(int numLayers, int *layerSizes, double learningRate) {
    
    MLP *net = malloc(sizeof(MLP));
    net->numLayers = numLayers;
    net->layerSizes = malloc(numLayers * sizeof(int));

    // Alloc ** ptrs that contain all layers
    net->weights = malloc((numLayers - 1) * sizeof(double *));
    net->biases = malloc((numLayers - 1) * sizeof(double *));
    // NOTE Initialize outputs the same way (this will store fact(weights*input+biases))
    net->outputs = malloc((numLayers - 1) * sizeof(double *));
    net->dOutputs = malloc((numLayers - 1) * sizeof(double *));
    net->matprod = malloc((numLayers - 1) * sizeof(double *));
    net->inputAdjoints = malloc((numLayers - 1) * sizeof(double *));
 
    net->learningRate = learningRate;

    srand(time(NULL));

    // Alloc individual layers
    for (int i = 0; i < numLayers - 1; i++) {
        int rows = layerSizes[i + 1];
        int cols = layerSizes[i];
        net->weights[i] = malloc(rows * cols * sizeof(double));
        net->biases[i] = malloc(rows * sizeof(double));
        // Outputs have the same size as biases
        net->outputs[i] = malloc(rows * sizeof(double));
        net->dOutputs[i] = malloc(rows * sizeof(double));
        net->matprod[i] = malloc(rows * sizeof(double));
        net->inputAdjoints[i] = malloc(cols * sizeof(double));

        // Initialize weights and biases (random weights, biases set to zero)
        for (int j = 0; j < rows * cols; j++) {
            net->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Weights between -1 and 1
        }
        for (int j = 0; j < rows; j++) {
            net->biases[i][j] = 0; // Biases initialized to 0
        }
    }

    // Copy layers sizes
    for (int i = 0; i < numLayers; i++) {
        net->layerSizes[i] = layerSizes[i];
    }

    return net;
}

void main() {
    return 0;
}