#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "../matrix/matrix.h"

double sigmoid(double x);
void apply_function(Matrix* m, double (*func)(double));
void backpropagate(Matrix* input, Matrix* output, Matrix* expected, Matrix* weights, Matrix* biases, double learning_rate);
void softmax(Matrix* m);

#endif // NEURAL_NETWORK_H
