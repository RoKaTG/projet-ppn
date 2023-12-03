#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdint.h>
#include "../matrixOperand/matrixOperand.h"

double sigmoid(double x);
void apply_function(Matrix* m, double (*func)(double));
void initialize_weights_and_biases(Matrix* matrix);

#endif // NEURAL_NETWORK_H
