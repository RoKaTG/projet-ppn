#include <cblas.h>
#include <clapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

#include "../include/networks/mlp_blas.h"
#include "../include/networks/activation.h"
#include "../include/mnist_reader/mnist_reader.h"
#include "../include/benchmark/bench.h"
 
/**************************************/
/*                                    */
/*          Main function             */
/*          Initialization            */
/*          Training phase            */
/*          Testing phase             */
/*          Benchmarking              */
/*                                    */
/**************************************/

int main(int argc, char *argv[]) {
    if (argc < 2 || (strcmp(argv[1], "true") != 0 && strcmp(argv[1], "false") != 0)) {
        printf("Usage : %s [routine] [Topology] [Activation] [TrainingSample] [numEpoch] [batchSize]\n", argv[0]);
        printf("                ^          ^         ^              ^                         ^         \n");
        printf("                |          |         |              |                         |         \n");
        printf("batch:true   ___|          |         |              |__=< 60000               |___only when routine:true\n");
        printf("classic:false              |         |                                                  \n");
        printf("                           |         |                                                  \n");
        printf("Separate layers by ','_____|         |___ relu || sigmoid || soft                       \n");
            
        return 1;
    }

    if (strcmp(argv[1], "true") == 0) {
        if (argc != 7) {
            printf("Usage : %s [routine] [Topology] [Activation] [TrainingSample] [numEpoch] [batchSize]\n", argv[0]);
            printf("                ^          ^         ^              ^                         ^         \n");
            printf("                |          |         |              |                         |         \n");
            printf("batch:true   ___|          |         |              |__=< 60000               |___ =< 64\n");
            printf("classic:false              |         |                                                  \n");
            printf("                           |         |                                                  \n");
            printf("Separate layers by ','_____|         |___ relu || sigmoid || soft                       \n");
            
            return 1;
        }
    }

    if (strcmp(argv[1], "false") == 0) {
        if (argc != 6) {
            printf("Usage : %s [routine] [Topology] [Activation] [TrainingSample] [numEpoch]\n", argv[0]);
            printf("                ^          ^         ^              ^                   \n");
            printf("                |          |         |              |                   \n");
            printf("batch:true   ___|          |         |              |__=< 60000         \n");
            printf("classic:false              |         |                                  \n");
            printf("                           |         |                                  \n");
            printf("Separate layers by ','_____|         |___ relu || sigmoid || tanh       \n");
            
            return 1;
        }
    }

    double start_t, finish_t, exec_t;
    Benchmark result;

    bool routine = (strcmp(argv[1], "true") == 0) ? true : false;

    if (routine != false && routine !=true) printf("Error: You have to specify the training routine.\n");

    int numLayers = 1;
    for (char *p = argv[2]; *p; p++) numLayers += (*p == ',');
    int *layerSizes = malloc(numLayers * sizeof(int));
    char *token = strtok(argv[2], ",");
    for (int i = 0; i < numLayers && token != NULL; i++) {
        layerSizes[i] = atoi(token);
        token = strtok(NULL, ",");
    }

    // NOTE After filling layerSizes array from command line arguments
    if (layerSizes[0] != 784 || layerSizes[numLayers - 1] != 10) {
        printf("Error: The first layer size must be 784 and the last layer size must be 10.\n");
        free(layerSizes);
        return 1;
    }

    char *func = argv[3];

    if (strcmp(func, "relu") != 0 && strcmp(func, "sigmoid") != 0 && strcmp(func, "tanh") != 0) {
        printf("Error: The activation function must be either relu OR sigmoid OR tanh.\n");
        
        return 1;    
    }

    int activation;

    if (strcmp(func, "relu") == 0) activation = 1;
    if (strcmp(func, "sigmoid") == 0) activation = 2;
    if (strcmp(func, "tanh") == 0) activation = 3;

    int numTrainingImages = atoi(argv[4]);

    if (numTrainingImages > 60000) {
        printf("Error: the training sample can't be superior to 60 000.\n");
       
        return 1;
    }

    int numEpochs = atoi(argv[5]);

    if (numEpochs >= 30) {
        printf("Warning: The number of epochs exceeds 30, that may be time & ressource consuming.\n");
    }

    int batchSize;// = atoi(argv[6]);

    double learningRate = 0.01; // NOTE Learning rate being set at 10^-2 (will be decaying in a future update)

    int numTestImages = 10000; 

    double lambda = 0.001;
    
    printf("Network topology:");
    for (int i = 0; i < numLayers; i++) {
        if (i < numLayers - 1) {
            printf(" Layer %d: %d neurons |", i, layerSizes[i]);
        } else {
            printf(" Layer %d: %d neurons.", i, layerSizes[i]);
        }
    }

    printf("\n\n");

    
    MLP *net = create_mlp(numLayers, layerSizes, learningRate);
    
    if (routine == true) {
        batchSize = atoi(argv[6]);
        if (numTrainingImages % batchSize != 0) {
            printf("Error: Your batch's size has to be a divisor of your training sample.\n");

            return 1;
        }
        start_t = omp_get_wtime();
        trainBatch(net, numTrainingImages, batchSize, numEpochs, lambda, activation);
        finish_t = omp_get_wtime();
    } else {
        start_t = omp_get_wtime();
        trainMLP(net, &result, numEpochs, numTrainingImages, lambda, activation);
        finish_t = omp_get_wtime();
    }

    exec_t = finish_t - start_t;

    printf("\n\n");
    // Testing the network after the training session (same methodology)
	float res = testMLP(net, numTestImages, activation);
    //Printing the MLP's accuracy & time execution
    printf("Time execution: %lfs\n", exec_t);
    printf("Accuracy: %.2f%% | Loss: %.2f%%\n", res, 100 - res);

    char* topologyStr = malloc(1024 * sizeof(char));
    strcpy(topologyStr, "");

    for(int i = 0; i < numLayers; i++) {
        char layerSizeStr[20]; // Pour convertir la taille de la couche en chaîne
        sprintf(layerSizeStr, "%d", layerSizes[i]);
        strcat(topologyStr, layerSizeStr);

        if (i < numLayers - 1) {
            strcat(topologyStr, ",");
        }
    }

    result.routine = routine ? "batch" : "default";
    result.actFunction = func;
    result.topology = topologyStr;
    result.trainingImages = numTrainingImages;
    result.epochs = numEpochs;
    result.batchSize = routine ? batchSize : -1;
    result.precision = res;
    result.totalTime = exec_t;
    result.errorRate = 100.0f - res;

    printf("%-10s %-20s %-10s %-20s %-10s %-15s %-15s %-20s %-20s %-15s\n", 
        "Routine", "Topology", "Act Func", "Training Images", "Epochs", 
        "Batch Size", "Precision (%)", "Total Time (s)", "Avg Epoch Time (s)", "Error Rate (%)");

    printBenchmarkResult(&result);

    free_mlp(net);

    return 0;
}
