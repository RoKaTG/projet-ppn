#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../mnist_reader/mnist_reader.h"
#include "neural_network.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
