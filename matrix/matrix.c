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

void print_matrix(Matrix *matrix) {
    if (matrix == NULL) {
        printf("Matrix is NULL\n");
        return;
    }

    printf("This is the content of this %d x %d matrix:\n", matrix->row, matrix->column);
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->column; j++) {
            printf("%lf \t", matrix->value[i][j]);
        }
        printf("\n");
    }
}
