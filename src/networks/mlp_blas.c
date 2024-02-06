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

/**************************************/
/*                                    */
/*       Feedforward application      */
/*                                    */
/**************************************/

// NOTE We also need the squared norm and its derivative

/**
 * Compute the squared norm of a vector.
 *
 * @param x The input vector.
 * @param n The number of elements in the vector.
 * @return The squared norm of the vector.
 */
double squaredNorm(double *x, int n) {
    double sum = 0.;
    for(int i=0; i<n; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

/**
 * Compute the derivative of the squared norm with respect to the input vector.
 *
 * @param x  The input vector.
 * @param dx The output vector containing the derivatives.
 * @param n  The number of elements in the vector.
 */
void squaredNormPrime(double *x, double *dx, int n) {
    for(int i=0; i<n; i++) {
        dx[i] = 2 * x[i];
    }
}

/**
 * Perform the feedforward operation for a Multilayer Perceptron (MLP) neural network.
 * Compute the network's output given an input and compare it to the expected output.
 *
 * @param net The MLP network to perform the feedforward operation on.
 * @param input The input data for the network.
 * @param expected The expected output data for comparison.
 * @return The calculated error (delta) between the network's output and the expected output.
 */
int feedforward(MLP *net, double *input, double *expected) {
    printf("Starting feedforward, checking pointers...\n");
    printf("net: %p, net->outputs[0]: %p\n", (void *)net, (void *)net->outputs[0]);

    // NOTE This will point to the current layer input. Note how we are not copying any memory.
    double *layerInput = input;

    // Calculate the output for each subsequent layer
    // NOTE Indices start at 0 to make things clearer
    for (int i = 0; i < net->numLayers - 1; i++) {
        int M = net->layerSizes[i + 1];   // Number of rows in the weight matrix (and output size)
        int N = 1;                        // Since the input is a vector
        int K = net->layerSizes[i];       // Number of columns in the weight matrix (and input size)

        // Perform matrix multiplication
        // NOTE We are doing everything on the same layer, so we are indexing all matrices
        // using simply i.
        // Also, be careful, we need to set beta to 0 (we overwrite C/matprod completely)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, 
                    net->weights[i], K, layerInput, N, 0.0, net->matprod[i], N);

        // Apply the activation function (sigmoid) to each element of net->outputs[i]
        // NOTE Don't forget to apply biases (we do it in the same loop)
        // Also, in the future, it should be faster to move this for loop inside the sigmoid function.
        // This way, we will be performing only 1 function call (vs M currently)
        for (int j = 0; j < M; j++) {
            net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
            // NOTE Store activation function (sigmoid) derivative
            net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
        }

        // NOTE Set input for the next layer i+1
        layerInput = net->outputs[i];
    }

    // NOTE From here, layerInput actually points to the last layer output -> we'll reuse it to compute 
    // the network's delta and the error's norm
    double *netOutput = layerInput;

    // NOTE Compute the cost
    for (int i = 0; i < net->layerSizes[net->numLayers - 1]; i++) {
        net->deltas[i] = expected[i] - netOutput[i];
    }
    // NOTE Computing squaredNorm is technically useless for the backward, but we'll do it 
    // anyway because it is cheap, and it can be interesting to study the evolution of this value over 
    // the training phase
    net->delta = squaredNorm(&(net->deltas[0]), net->layerSizes[net->numLayers - 1]);
    // NOTE However, we do need the squaredNormPrime for the backward
    squaredNormPrime(&(net->deltas[0]), &(net->dDeltas[0]), net->layerSizes[net->numLayers - 1]);
    
    return net->delta;
}

void main() {
    return 0;
}