#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../matrix/matrix.h"
#include "../matrix_operand/matrixOperand.h"
#include "../mnist_reader/mnist_reader.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void apply_function(Matrix* m, double (*func)(double)) {
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->column; j++) {
            m->value[i][j] = func(m->value[i][j]);
        }
    }
}

void backpropagate(Matrix* input, Matrix* output, Matrix* expected, Matrix* weights, Matrix* biases, double learning_rate) {
    double error = expected->value[0][0] - output->value[0][0];
    double derivative = output->value[0][0] * (1 - output->value[0][0]);
    double gradient = error * derivative;
    
    for (int i = 0; i < weights->row; i++) {
        for (int j = 0; j < weights->column; j++) {
            weights->value[i][j] += learning_rate * gradient * input->value[j][0];
        }
        biases->value[i][0] += learning_rate * gradient;
    }
}
