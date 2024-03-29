#include <cblas.h>
#include <clapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "../../include/networks/mlp_blas.h"
#include "../../include/networks/activation.h"
#include "../../include/mnist_reader/mnist_reader.h"
#include "../../include/benchmark/bench.h"
 
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

    // NOTE After initializing weights and biases
    net->weightGradients = malloc((numLayers - 1) * sizeof(double *));
    net->biasGradients = malloc((numLayers - 1) * sizeof(double *));

    for (int i = 0; i < numLayers - 1; i++) {
        int rows = layerSizes[i + 1];
        int cols = layerSizes[i];
        net->weightGradients[i] = malloc(rows * cols * sizeof(double));
        net->biasGradients[i] = malloc(rows * sizeof(double));

        // Initialize gradients to zero
        for (int j = 0; j < rows * cols; j++) {
            net->weightGradients[i][j] = 0.0;
        }
        for (int j = 0; j < rows; j++) {
            net->biasGradients[i][j] = 0.0;
        }
    }

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
int feedforward(MLP *net, double *input, double *expected, int activation) {
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

        // Apply the activation function to each element of net->outputs[i] & apply softmax || sigmoid to the last layer
        // NOTE Don't forget to apply biases (we do it in the same loop)
        // Also, in the future, it should be faster to move this for loop inside the sigmoid function.
        // This way, we will be performing only 1 function call (vs M currently)
            if (activation == 1) {
                if (i == net->numLayers - 2) {
                    for (int j = 0; j < M; j++) {
                        net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                        net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
                    }
                } else {
                    for (int j = 0; j < M; j++) {
                        net->outputs[i][j] = relu(net->matprod[i][j] + net->biases[i][j]);   
                        // NOTE Store activation function (relu) derivative
                         net->dOutputs[i][j] = reluPrime(net->matprod[i][j] + net->biases[i][j]);
                    }
                }
            }

            if (activation == 2) {
                if (i == net->numLayers - 2) {
                    softmax(net->matprod[i], net->outputs[i], M);
                    softmax(net->outputs[i], net->dOutputs[i], M); // NOTE The way we implemented softmax make it that
                } else {                                           // softmax = softmaxPrime so we store derivative this way
                    for (int j = 0; j < M; j++) {
                        net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                        // NOTE Store activation function (sigmoid) derivative
                        net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
                    }
                }                                                
            }

            /*for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                // NOTE Store activation function (sigmoid) derivative
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }*/
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
void backpropagate(MLP *net, double *netInput, double lambda) {
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

        for (int j = 0; j < M * K; j++) {
            // NOTE we apply L2 regularization with weight decay
            net->weights[i][j] -= lambda * net->weights[i][j] * net->learningRate;
        }
        // Also, note how we perform the DGEMMs in this particular order, as we overwrite biases

        // Finally, set the prevInput to the current inputAdjoint
        prevInput = net->inputAdjoints[i];
    }
}

/*****************************************/
/*                                       */
/* Training & Prediction & Testing phase */
/*                                       */
/*****************************************/

/**
 * Train the MLP network on a set of inputs and targets.
 * Perform forward propagation and backpropagation to update the weights.
 *
 * @param net A pointer to the MLP network.
 * @param input An array of inputs for training.
 * @param target An array of target outputs for training.
 */
void trainMLP(MLP *net, Benchmark *result, int numEpochs, int numTrainingImages, double lambda, int activation) {
    // Paths to data files
    const char *imageFilePath = "data/train-images-idx3-ubyte";
    const char *labelFilePath = "data/train-labels-idx1-ubyte";

    // Opening data files
    FILE *imageFile = fopen(imageFilePath, "rb");
    FILE *labelFile = fopen(labelFilePath, "rb");
    if (!imageFile || !labelFile) {
        printf("Error opening data files.\n");
        return;
    }

    // Reading training data
    uint8_t *images = readMnistImages(imageFile, 0, numTrainingImages);
    uint8_t *labels = readMnistLabels(labelFile, 0, numTrainingImages);

    double start_t, end_t, exec_t;

    double totalExecTime = 0.0;

    // Training cycle
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        start_t = omp_get_wtime();
        for (int i = 0; i < numTrainingImages; i++) {
            double input[784];
            double target[10] = {0};

            // Normalization and One-hot encoding
            for (int j = 0; j < 784; j++) {
                input[j] = images[i * 784 + j] / 255.0;
            }
            target[labels[i]] = 1.0;

            // Forward and backward propagation
            feedforward(net, input, target, activation);
            backpropagate(net, input, lambda);
        }
        end_t = omp_get_wtime();
        exec_t = end_t - start_t;
        totalExecTime += exec_t;
        printf("Epoch %d/%d completed using default routine in %lfs.\n", epoch + 1, numEpochs, exec_t);
    }
    result->avgEpochTime = totalExecTime / numEpochs; // Computing the mean time execution

    // Cleanup
    fclose(imageFile);
    fclose(labelFile);
    free(images);
    free(labels);
}


/**
 * Predict the output of the MLP network for a given input.
 * Use forward propagation to calculate the output.
 *
 * @param net A pointer to the MLP network.
 * @param input An array of inputs for prediction.
 * @return A pointer to the predicted output.
 */
double *predict(MLP *net, double *input, int activation) {
    // NOTE Because we don't need to compute or store any partial derivatives during the 
    // prediction phase, The returned pointer will simply point to the last element of 
    // "outputs" in the MLP structure

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
        if (activation == 1) {
            if (i == net->numLayers - 2) {
                for (int j = 0; j < M; j++) {
                    net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                }
            } else {
                for (int j = 0; j < M; j++) {
                    net ->outputs[i][j] = relu(net->matprod[i][j] + net->biases[i][j]);
                }
            }
        }

        if (activation == 2) {
            if (i == net->numLayers - 2) {
                softmax(net->matprod[i], net->outputs[i], M);
            } else {
                for (int j = 0; j < M; j++) {
                    net ->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                }
            }
        }

        // NOTE Set input for the next layer i+1
        layerInput = net->outputs[i];
    }

    // NOTE From here, layerInput points to the last layer output -> we'll re-use it to compute 
    // the network's delta and the error's norm
    double *netOutput = layerInput;
    return netOutput;
}

/**
 * Test the MLP network using a dataset.
 * Reads test images and labels from files and evaluates the network's performance.
 *
 * @param net A pointer to the MLP network.
 * @param numTestImages Number of test images to evaluate.
 * @return The accuracy of the network's predictions on the test dataset.
 */
double testMLP(MLP *net, int numTestImages, int activation) {
    FILE *testImageFile = fopen("data/t10k-images-idx3-ubyte", "rb");
    FILE *testLabelFile = fopen("data/t10k-labels-idx1-ubyte", "rb");
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

        output = predict(net,input, activation);

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
        /*
        // Printing the vector of prediction for each image
        printf("Prediction for image %d: [", i);
        for (int j = 0; j < 10; j++) {
            if (j != 9) {
                printf("%f, ", output[j]);
            }
            if (j == 9) {
                printf("%f] - Real Label: %d\n", output[j], testLabels[i]); 
            }
        }*/

    }
    // Calculation of accuracy
    double accuracy = 100.0 * correctPredictions / numTestImages;
    
    fclose(testImageFile);
    fclose(testLabelFile);
    free(testImages);
    free(testLabels);

    return accuracy;
}

/*****************************************/
/*                                       */
/*           Batch functions             */
/*                                       */
/*****************************************/

void batching(MLP *net, double **inputs, double **targets, int batchSize, double lambda, int activation) {
    // Reset gradient accumulators to zero for weights and biases
    for (int i = 0; i < net->numLayers - 1; i++) {
        memset(net->weightGradients[i], 0, net->layerSizes[i + 1] * net->layerSizes[i] * sizeof(double));
        memset(net->biasGradients[i], 0, net->layerSizes[i + 1] * sizeof(double));
    }

    // Accumulate gradients for each input in the batch
    for (int b = 0; b < batchSize; b++) {
        // NOTE We can use the feedforward to compute outputs and derivative outputs because we compute them in here to ease the batching process
        feedforward(net, inputs[b], targets[b], activation); 

        // NOTE We do a backpropagate & use the errors and accumulate gradients but without updating weights and biases
        double *prevDelta = calloc(net->layerSizes[net->numLayers - 1], sizeof(double));
        for (int i = net->numLayers - 2; i >= 0; i--) {
            int M = net->layerSizes[i + 1];
            int N = net->layerSizes[i];
            double *delta = calloc(M, sizeof(double));

            // Compute delta for current layer
            if (i == net->numLayers - 2) { // Output layer
                for (int j = 0; j < M; j++) {
                    delta[j] = (net->outputs[i][j] - targets[b][j]) * net->dOutputs[i][j];
                }
            } else { // Hidden layers
                for (int j = 0; j < M; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < net->layerSizes[i + 2]; k++) {
                        sum += prevDelta[k] * net->weights[i + 1][k * M + j];
                    }
                    delta[j] = sum * net->dOutputs[i][j];
                }
            }

            // Accumulate gradients for weights and biases
            for (int j = 0; j < M; j++) {
                net->biasGradients[i][j] += delta[j]; // NOTE  We accumulate bias gradient this way
                for (int k = 0; k < N; k++) {
                    net->weightGradients[i][j * N + k] += delta[j] * (i > 0 ? net->outputs[i - 1][k] : inputs[b][k]); // NOTE This way is a bit complicated but is more efficent for weight gradient
                }
            }

            if (i < net->numLayers - 2) free(prevDelta);
            prevDelta = delta;
        }
        free(prevDelta);
    }

    // NOTE We use average gradients to apply updates with L2 regularization
    for (int i = 0; i < net->numLayers - 1; i++) {
        int M = net->layerSizes[i + 1];
        int N = net->layerSizes[i];
        for (int j = 0; j < M; j++) {
            net->biases[i][j] -= net->learningRate * net->biasGradients[i][j] / batchSize;
            for (int k = 0; k < N; k++) {
                double weightGradientAvg = net->weightGradients[i][j * N + k] / batchSize;
                net->weights[i][j * N + k] -= net->learningRate * (weightGradientAvg + lambda * net->weights[i][j * N + k]);
            }
        }
    }
}

/**
 * Train the MLP network using mini-batch gradient descent.
 *
 * @param net A pointer to the MLP network.
 * @param numTrainingImages Total number of training images.
 * @param batchSize Size of each mini-batch.
 * @param numEpochs Number of epochs for training.
 * @param lambda Regularization parameter for weight decay.
 */
void trainBatch(MLP *net, Benchmark *result, int numTrainingImages, int batchSize, int numEpochs, double lambda, int activation) {
    const char *imagePath = "data/train-images-idx3-ubyte";
    const char *labelPath = "data/train-labels-idx1-ubyte";

    FILE *imageFile = fopen(imagePath, "rb");
    FILE *labelFile = fopen(labelPath, "rb");

    uint8_t *images = readMnistImages(imageFile, 0, numTrainingImages);
    uint8_t *labels = readMnistLabels(labelFile, 0, numTrainingImages);

    double *inputBatch[batchSize];
    double *targetBatch[batchSize];

    int numBatches = numTrainingImages / batchSize;

    double start_t, end_t, exec_t;

    double totalExecTime = 0.0;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
        start_t = omp_get_wtime();
        for (int batch = 0; batch < numBatches; batch++) {
            for (int i = 0; i < batchSize; i++) {
                int imageIndex = batch * batchSize + i;
                inputBatch[i] = (double *)malloc(784 * sizeof(double));
                targetBatch[i] = (double *)calloc(10, sizeof(double));

                for (int j = 0; j < 784; j++) {
                    inputBatch[i][j] = images[imageIndex * 784 + j] / 255.0;
                }
                targetBatch[i][labels[imageIndex]] = 1.0;
            }
            batching(net, inputBatch, targetBatch, batchSize, lambda, activation);

            for (int i = 0; i < batchSize; i++) {
                free(inputBatch[i]);
                free(targetBatch[i]);
            }
        }
        end_t = omp_get_wtime();
        exec_t = end_t - start_t;
        totalExecTime += exec_t;
        printf("Epoch %d/%d completed using mini batches routine in %lfs.\n", epoch + 1, numEpochs, exec_t);
    }
    result->avgEpochTime = totalExecTime / numEpochs; // Computing the mean time execution

    fclose(imageFile);
    fclose(labelFile);
    free(images);
    free(labels);
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