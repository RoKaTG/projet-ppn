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
    return exp(x) / pow((exp(-x) + 1), 2);
}