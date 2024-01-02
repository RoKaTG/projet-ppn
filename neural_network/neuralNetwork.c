#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#include "neuralNetwork.h"
#include "../mnist_reader/mnist_reader.h"
#include "../matrix_operand/matrixOperand.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

int main() {
    return 0;
}
