#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../mnist_reader/mnist_reader.h"
#include "neural_network.h"

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
