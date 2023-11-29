#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix.h"


Matrix* create_matrix(int rows, int columns) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (matrix != NULL) {
        matrix->values = (double**)malloc(sizeof(double*));
        matrix->rows = rows;
        matrix -> columns = columns;
        if (matrix->values != NULL) {
            for(int i = 0; i < row; i++) {
                matrix->values[i] = (double*)malloc(col * sizeof(double));
            }
        }
    }
    return matrix;
}

int fill_matrix(Matrix *matrix, double values[], int values_size) {
    if (matrix == NULL || values == NULL) {
        return -1;
    }

    int needed_size = matrix->row * matrix->column;
    if (values_size < needed_size) {
        return -1;
    }

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            matrix->value[i][j] = values[i * matrix->column + j];
        }
    }
    return 0;
}
