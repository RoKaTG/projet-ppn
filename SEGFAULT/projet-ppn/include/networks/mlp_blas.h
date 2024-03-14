#pragma once

#include "../benchmark/bench.h"

/**************************************/
/*                                    */
/*          Network stucture          */
/*                                    */
/**************************************/

typedef struct {
    int numLayers;
    int *layerSizes;

    // NOTE This is the useful stuff for the FORWARD propagation
    double **weights;
    double **biases;
    double **outputs;
    // NOTE We only need one delta for the last layer
    // We set its size to a constant 10, but we know this is won't work in the general case
    double deltas[10]; // Contains actual - ideal outputs
    double delta; // Norm of deltas -> Scalar cost & final network output

    // NOTE This is the useful stuff for the BACKWARD propagation.
    // However, we will compute some of those values during the forward and reuse them later
    // Also, we don't need to store all the intermediate values and adjoints
    // (See backpropagate function to understand why we only need what's below)
    double **dOutputs;
    double **matprod;
    double **inputAdjoints;
    double dDeltas[10];

    double learningRate;

    // NOTE We need gradient accumulators for batch processing
    double **weightGradients;  // Accumulators for weight gradients
    double **biasGradients;    // Accumulators for bias gradients
} MLP;

/**************************************/
/*                                    */
/*          Network's headers         */
/*                                    */
/**************************************/

MLP* create_mlp(int numLayers, int *layerSizes, double learningRate);

/**************************************/
/*                                    */
/*        Feedforward's headers       */
/*                                    */
/**************************************/

double squaredNorm(double *x, int n);
int feedforward(MLP *net, double *input, double *target, int activation);

/**************************************/
/*                                    */
/*        Backward's headers          */
/*                                    */
/**************************************/

void squaredNormPrime(double *x, double *dx, int n);
void backpropagate(MLP *net, double *netInput, double lambda);

/***************************************************/
/*                                                 */
/*  Training's & Prediction's & Testing's headers  */
/*                                                 */
/***************************************************/

double *predict(MLP *net, double *input, int activation);
void trainMLP(MLP *net, Benchmark *result, int numEpochs, int numTrainingImages, double lambda, int activation);
double testMLP(MLP *net, int numTestImages, int activation);

/**************************************/
/*                                    */
/*     Memory management header       */
/*                                    */
/**************************************/

void free_mlp(MLP *net);

/**************************************/
/*                                    */
/*           Batch headers            */
/*                                    */
/**************************************/

void batching(MLP *net, double **inputs, double **targets, int batchSize, double lambda, int activation);
void trainBatch(MLP *net, Benchmark *result, int numTrainingImages, int batchSize, int numEpochs, double lambda, int activation);