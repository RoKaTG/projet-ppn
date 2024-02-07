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
#include "../../include/mnist_reader/mnist_reader.h"
 
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

/**************************************/
/*                                    */
/*       Backward application         */
/*                                    */
/**************************************/

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
 * Perform backpropagation on the MLP network.
 * Update the weights and biases based on the error computed with respect to the target output.
 *
 * @param net A pointer to the MLP network.
 * @param netInput The input data for the network.
 */
void backpropagate(MLP *net, double *netInput) {
    int lastLayerIndex = net->numLayers - 2; // NOTE This used to be "-1".

    // NOTE Computing the cost was moved to the forward propagation
    // Here, we will only apply the chain rule
    
    // NOTE Implicit chain rule application for the cost computation: 
    //   zbar = ||c||bar = 1; minusdot=dDeltas in code; factdot=(1,...,1)
    //   -> minusbar = minusdot*zbar;
    //   -> factbar = factdot*minusbar = minusbar
    // Thus, dDeltas already contains the adjoint for the last layer's output, and the next 
    // operation is the last layer's activation function

    // Propagate the error backward through the previous layers and update the weights and biases
    // NOTE i>= is important here, to make sure we visit the last layer
    double *prevInput = &(net->dDeltas[0]);
    for (int i = lastLayerIndex; i >= 0; i--) {
        int layerSize = net->layerSizes[i + 1];

        int M = net->layerSizes[i + 1];     // Number of rows in the weight matrix (and output size)
        int N = 1;                        // Since the input is a vector
        int K = net->layerSizes[i];       // Number of columns in the weight matrix (and input size)

        // plusbar = plusdot * lk+1bar (lk+1bar denotes the last output adkoint, prevInput in the code)
        for (int j = 0; j < layerSize; j++) {
            // NOTE We can overwrite dOutputs as we will not need it anymore.
            // This avoids allocating temporary arrays to store the "current result"
            net->dOutputs[i][j] = net->dOutputs[i][j] * prevInput[j];
        }

        // bbar = bdot * plusbar; bdot = 1
        // Thus, bbar = bdot, and we can directly apply corrections to biases
        for (int j = 0; j < layerSize; j++) {
            net->biases[i][j] -= net->dOutputs[i][j] * net->learningRate;
        }

        // timesbar = timesdot * plusbar; timesdot = 1

        // lkbar = lkdot * timesbar; This is one of the matprod special cases.
        // We will compute the current layer input (= the previous layer output)
        // This will be used to continue backprop for the previous layer
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0, 
                    net->weights[i], K, net->dOutputs[i], N, 0.0, net->inputAdjoints[i], N);

        // wbar = wdot * timesbar; This is one of the matprod special cases.
        // We'll compute the matprod between the previous output's transpose and timesbar
        // We will directly apply the correction using the DGEMM 
        // (by setting alpha to net->learningRate, and beta to 1)
        double *matprodInput = NULL;
        if (i > 0) {
            matprodInput = net->outputs[i - 1];
        }
        else {
            // Be careful, if i == 0, the previous layer's output is not defined.
            // It will actually be the network's output
            matprodInput = netInput;
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, net->learningRate, 
                    net->dOutputs[i], N, matprodInput, N, 1.0, net->weights[i], K);

        // Also, note how we perform the DGEMMs in this particular order, as we overwrite biases

        // Finally, set the prevInput to the current inputAdjoint
        prevInput = net->inputAdjoints[i];
    }
}

/**************************************/
/*                                    */
/*    Training & Prediction phase     */
/*                                    */
/**************************************/

/**
 * Train the MLP network on a set of inputs and targets.
 * Perform forward propagation and backpropagation to update the weights.
 *
 * @param net A pointer to the MLP network.
 * @param input An array of inputs for training.
 * @param target An array of target outputs for training.
 */
void train(MLP *net, double *input, double *target) {
    // Feedforward
    feedforward(net, input, target); // NOTE feedforward returns the scalar cost

    // NOTE We  consider that computing the cost (output - target) is part of the feedworward
    // Backpropagation should only apply the chain rule

    // Backpropagation
    backpropagate(net, input);
}

/**
 * Predict the output of the MLP network for a given input.
 * Use forward propagation to calculate the output.
 *
 * @param net A pointer to the MLP network.
 * @param input An array of inputs for prediction.
 * @return A pointer to the predicted output.
 */
// NOTE Because we don't need to compute or store any partial derivatives during the 
// prediction phase, The returned pointer will simply point to the last element of 
// "outputs" in the MLP structure

double * predict(MLP *net, double *input) {
    // NOTE This will point to the current layer input. Note how we are not copying any memory.
    double *layerInput = input;

    // Compute the output for each subsequent layer
    // NOTE Indices start at 0 to make things clearer
    for (int i = 0; i < net->numLayers - 1; i++) {
        int M = net->layerSizes[i + 1];     // Number of rows in the weight matrix (and output size)
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
        }

        // NOTE Set input for the next layer i+1
        layerInput = net->outputs[i];
    }

    // NOTE From here, layerInput actually points to the last layer output -> we'll reuse it to compute 
    // the network's delta and the error's norm
    double *netOutput = layerInput;
    return netOutput;
}

/**************************************/
/*                                    */
/*         Memory management          */
/*                                    */
/**************************************/

/**
 * Deallocate memory for the MLP network.
 * Cleans up all allocated weights, biases, and structures.
 *
 * @param net A pointer to the MLP network to free.
 */
void free_mlp(MLP *net) {
    if (net != NULL) {
        // NOTE We now have the same numbers of weight, biases, outputs
        // Also, we do not need to free deltas (now a static array)
        for (int i = 0; i < net->numLayers - 1; i++) {
            free(net->weights[i]);
            free(net->biases[i]);
            free(net->outputs[i]);
            free(net->dOutputs[i]);
            free(net->matprod[i]);
            free(net->inputAdjoints[i]);
        }

        free(net->weights);
        free(net->biases);
        free(net->outputs);
        free(net->dOutputs);
        free(net->inputAdjoints);
        free(net->matprod);
        free(net->layerSizes);
        free(net);
    }
}

/**************************************/
/*                                    */
/*       Main function used for       */
/*          Initializing the MLP      */
/*          Training phase            */
/*          Testing phase             */
/*  Print accuracy & execution time   */
/*                                    */
/**************************************/

int main() {
    // Network's initialization
    int layerSizes[] = {784, 300, 10}; // 1 hidden layer of size : 300
    double learningRate = 0.01; // Learning rate being set at 10^-2 (will be decaying in a future update)
    int numLayers = sizeof(layerSizes) / sizeof(layerSizes[0]);
    MLP *net = create_mlp(numLayers, layerSizes, learningRate);

    // Opening & reading the mnist train sample
    FILE *imageFile = fopen("../../data/train-images-idx3-ubyte", "rb");
    FILE *labelFile = fopen("../../data/train-labels-idx1-ubyte", "rb");
    int numTrainingImages = 10000;   // Training sample
    uint8_t *images = readMnistImages(imageFile, 0, numTrainingImages);
    uint8_t *labels = readMnistLabels(labelFile, 0, numTrainingImages);

    int numEpochs = 5; // Number of epoch

    // Training cycle
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        for (int i = 0; i < numTrainingImages; i++) {

            double input[784];
            double target[10] = {0};

            // Normalization & One-hot encoding
            for (int j = 0; j < 784; j++) {
                input[j] = images[i * 784 + j] / 255.0;
            }
            target[labels[i]] = 1.0;

            train(net, input, target);
        }
    }

    // Testing the network after the training session (same methodology)
    FILE *testImageFile = fopen("../../data/t10k-images-idx3-ubyte", "rb");
    FILE *testLabelFile = fopen("../../data/t10k-labels-idx1-ubyte", "rb");
    int numTestImages = 10000;
    uint8_t *testImages = readMnistImages(testImageFile, 0, numTestImages);
    uint8_t *testLabels = readMnistLabels(testLabelFile, 0, numTestImages);

    int correctPredictions = 0;
    for (int i = 0; i < numTestImages; i++) {
        double input[784];
        double *output = NULL;

        // Normalization
        for (int j = 0; j < 784; j++) {
            input[j] = testImages[i * 784 + j] / 255.0;
        }

        output = predict(net, input);

        // Comparing label's prediction & ideal label
        int predictedLabel = 0;
        double maxOutput = output[0];
        for (int j = 1; j < 10; j++) {
            if (output[j] > maxOutput) {
                maxOutput = output[j];
                predictedLabel = j;
            }
        }

        if (predictedLabel == testLabels[i]) {
            correctPredictions++;
        }

        // Printing the vector of prediction for each images
        printf("Prédiction pour l'image %d : [", i);
        for (int j = 0; j < 10; j++) {
            printf("%f, ", output[j]);
        }
        printf("] - Label Réel : %d\n", testLabels[i]);    
        //printf("Prédiction pour l'image %d : [", i);
    }

    //Printing the MLP's accuracy
    printf("Précision: %.2f%%\n", 100.0 * correctPredictions / numTestImages);

    // Deallocate memory
    fclose(imageFile);
    fclose(labelFile);
    fclose(testImageFile);
    fclose(testLabelFile);
    free(images);
    free(labels);
    free(testImages);
    free(testLabels);
    free_mlp(net);

    return 0;
}