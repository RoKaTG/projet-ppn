#include "../../include/networks/feeding.h"
#include "../../include/networks/mlp_blas.h"
#include "../../include/networks/activation.h"


/**
 * Perform activation function on the hidden layer and output layer of a Multilayer Perceptron (MLP) network.
 *
 * This function applies different activation functions based on the specified activation code
 *
 * @param net Pointer to the MLP network.
 * @param activation Code representing the activation function combination to be applied.
 * @param i Index of the layer being processed.
 * @param M Number of elements in the layer.
 */
void feeding(MLP *net, int activation, int i, int M) {
    switch (activation)
    {
    // NOTE We apply relu on the hidden layer & sigmoid on the output layer
    case 1:               
        if (i != net->numLayers - 2) {                          
            relu_avx2(net->matprod[i], net->outputs[i], M);
            reluPrime_avx2(net->matprod[i], net->dOutputs[i], M);
        } else {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        }
        break;
    // NOTE We apply sigmoid on the hidden layer & softmax on the output layer
    case 2:
        if (i != net->numLayers - 2) {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        } else {
            softmax(net->matprod[i], net->outputs[i], M);
            softmax(net->outputs[i], net->dOutputs[i], M);
        }
        break;
    // NOTE We apply fast_sigmoid on the hidden layer & softmax on the output layer
    case 3:
        if (i != net->numLayers - 2) {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = fast_sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = fast_sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        } else {
            softmax(net->matprod[i], net->outputs[i], M);
            softmax(net->outputs[i], net->dOutputs[i], M);
        }
        break;
    // NOTE We apply LeakyRelu on the hidden layer & sigmoid on the output layer
    case 4:               
        if (i != net->numLayers - 2) {                          
            leakyReLU_avx2(net->matprod[i], net->outputs[i], M, 0.2);
            leakyReLUPrime_avx2(net->matprod[i], net->dOutputs[i], M, 0.2);
        } else {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        }
        break;
    // NOTE We apply tanh on the hidden layer & sigmoid on the output layer
    case 5:
        if (i != net->numLayers - 2) {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = tanhh(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = tanhPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        } else {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        }
        break;
    // NOTE We apply swish on the hidden layer & sigmoid on the output layer
        case 6:
        if (i != net->numLayers - 2) {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = swish(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = swishPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        } else {
            for (int j = 0; j < M; j++) {
                net->outputs[i][j] = sigmoid(net->matprod[i][j] + net->biases[i][j]);
                net->dOutputs[i][j] = sigmoidPrime(net->matprod[i][j] + net->biases[i][j]);
            }
        }
        break;
    }
}