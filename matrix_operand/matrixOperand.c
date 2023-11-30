#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrixOperand.h"

void scale_matrix(Matrix* matrix, double scalar) {
    if (matrix == NULL) return;

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            matrix->value[i][j] *= scalar;
        }
    }
}
