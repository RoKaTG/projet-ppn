#include <cblas.h>
#include <clapack.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

#include "../../include/networks/activation.h"

/**************************************/
/*                                    */
/*          Sigmoid's Function        */
/*                                    */
/**************************************/

/**
 * Sigmoid activation function.
 * Computes the sigmoid output of an input value.
 *
 * @param x The input value.
 * @return The output value of the sigmoid function.
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Derivative of the sigmoid activation function.
 * Used in computing the gradient during backpropagation.
 *
 * @param x The input value.
 * @return The derivative of the sigmoid function at that point.
 */
double sigmoidPrime(double x) {    
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

/**************************************/
/*                                    */
/*      Fast Sigmoid's Function       */
/*                                    */
/**************************************/

/**
 * Fast sigmoid function using double precision.
 * An approximation of the sigmoid function that is computationally faster.
 * It is defined as f(x) = x / (1 + |x|).
 *
 * @param x The input value (double).
 * @return The fast sigmoid of x (double).
 */
double fast_sigmoid(double x) {
    return x / (1.0 + fabs(x));
}

/**************************************/
/*                                    */
/*          Softmax's Function        */
/*                                    */
/**************************************/

/**
 * Apply the softmax function to the logits to compute probabilities.
 * Softmax function normalizes the logits and converts them into probabilities.
 * This prevents numerical stability issues.
 *
 * @param logits The input logits array.
 * @param probabilities The output probabilities array.
 * @param length The length of the logits and probabilities arrays.
 */
void softmax(double *logits, double *probabilities, int length) {
    double max_logit = -INFINITY; // Search for the maximum logit
    for (int i = 0; i < length; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        probabilities[i] = exp(logits[i] - max_logit); // Prevents numerical stability issues
        sum += probabilities[i];
    }
    
    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum;
    }
}

/**************************************/
/*                                    */
/*          reLu's Function           */
/*                                    */
/**************************************/

/**
 * Rectified Linear Unit (ReLU) activation function.
 *
 * @param x Input value.
 * @return Output value after applying ReLU activation.
 */
double relu(double x) {
    return (x > 0) ? x : 0;
}

/**
 * Derivative of the Rectified Linear Unit (ReLU) activation function.
 *
 * @param x Input value.
 * @return Derivative of the ReLU activation function at the given input.
 */
double reluPrime(double x) {
    return (x > 0) ? 1 : 0;
}

/**************************************/
/*                                    */
/*          tanh's Function           */
/*                                    */
/**************************************/
